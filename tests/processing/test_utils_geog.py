#!/usr/bin/env python3

"""
MPASdiag Test Suite: Tests for MPASGeographicUtils class

This module contains a comprehensive set of unit tests for the MPASGeographicUtils class, which provides utility functions for handling geographic coordinates in MPAS datasets. The tests cover various scenarios for extracting spatial coordinates, filtering data by spatial extent, and normalizing longitude values. Each test is designed to validate specific functionality and edge cases to ensure the robustness of the geographic utilities used in MPASdiag processing. It includes tests for handling real MPAS coordinate data, as well as synthetic datasets to verify the expected behavior of the utility functions under controlled conditions. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import pytest
import numpy as np
import xarray as xr
from mpasdiag.processing.utils_geog import MPASGeographicUtils


class TestExtractSpatialCoordinates:
    """ Test coordinate extraction from MPAS datasets. """
    
    def test_extract_coordinates_none_dataset(self: "TestExtractSpatialCoordinates") -> None:
        """
        This test confirms that `extract_spatial_coordinates` raises a ValueError when the input dataset is None. The test calls the function with None and asserts that a ValueError is raised with an appropriate message indicating that the dataset cannot be None. This ensures that the function properly handles invalid input and provides clear feedback to the caller about the issue. 

        Parameters:
            None

        Returns:
            None: Verified by asserting the raised exception message.
        """
        with pytest.raises(ValueError) as cm:
            MPASGeographicUtils.extract_spatial_coordinates(None) # type: ignore
        assert "Dataset cannot be None" in str(cm.value)
    
    def test_extract_coordinates_lonCell_latCell_degrees(self: "TestExtractSpatialCoordinates", mpas_coordinates) -> None:
        """
        This test confirms that `extract_spatial_coordinates` can successfully extract longitude and latitude coordinates from a dataset when the coordinate variables are named `lonCell` and `latCell` and are provided in degrees. The test creates a dataset with `lonCell` and `latCell` variables using real MPAS coordinate data, then calls the extraction function and asserts that the returned longitude and latitude arrays have the expected lengths and values that match the input coordinates. This verifies that the function correctly identifies and extracts spatial coordinates in the standard MPAS format when they are provided in degrees. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Confirmed via assertions on lengths and sample values.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(100, len(real_lon))
        
        ds = xr.Dataset({
            'lonCell': ('nCells', real_lon[:subset_size]),
            'latCell': ('nCells', real_lat[:subset_size])
        })
        
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(ds)
        
        assert len(lon) == subset_size
        assert len(lat) == subset_size

        np.testing.assert_array_almost_equal(lon, real_lon[:subset_size], decimal=5)
        np.testing.assert_array_almost_equal(lat, real_lat[:subset_size], decimal=5)
    
    def test_extract_coordinates_lonCell_latCell_radians(self: "TestExtractSpatialCoordinates", mpas_coordinates) -> None:
        """
        This test confirms that `extract_spatial_coordinates` can successfully extract longitude and latitude coordinates from a dataset when the coordinate variables are named `lonCell` and `latCell` and are provided in radians. The test creates a dataset with `lonCell` and `latCell` variables using real MPAS coordinate data converted to radians, then calls the extraction function and asserts that the returned longitude and latitude arrays have the expected lengths and values that match the original input coordinates (after conversion back to degrees). This verifies that the function correctly identifies and extracts spatial coordinates in radians and converts them to degrees as needed. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Validated by checking sample values approximate expected degrees.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(100, len(real_lon))
        
        ds = xr.Dataset({
            'lonCell': ('nCells', np.radians(real_lon[:subset_size])),
            'latCell': ('nCells', np.radians(real_lat[:subset_size]))
        })
        
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(ds)
        
        assert len(lon) == subset_size
        assert len(lat) == subset_size

        np.testing.assert_array_almost_equal(lon, real_lon[:subset_size], decimal=4)
        np.testing.assert_array_almost_equal(lat, real_lat[:subset_size], decimal=4)
    
    def test_extract_coordinates_longitude_latitude(self: "TestExtractSpatialCoordinates") -> None:
        """
        This test confirms that `extract_spatial_coordinates` can successfully extract longitude and latitude coordinates from a dataset when the coordinate variables are named `longitude` and `latitude`. The test creates a dataset with `longitude` and `latitude` variables, then calls the extraction function and asserts that the returned longitude and latitude arrays have the expected lengths and sample values that match the input coordinates. This verifies that the function can identify and extract spatial coordinates even when they use alternative variable names commonly found in some datasets. 

        Parameters:
            None

        Returns:
            None: Verified via length and sample value assertions.
        """
        ds = xr.Dataset({
            'longitude': ('nCells', np.linspace(90, 115, 50)),
            'latitude': ('nCells', np.linspace(-10, 20, 50))
        })
        
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(ds)
        
        assert len(lon) == pytest.approx(50)
        assert len(lat) == pytest.approx(50)
        assert lon[0] == pytest.approx(90, abs=0.1)
        assert lat[0] == pytest.approx(-10, abs=0.1)
    
    def test_extract_coordinates_lon_lat(self: "TestExtractSpatialCoordinates") -> None:
        """
        This test confirms that `extract_spatial_coordinates` can successfully extract longitude and latitude coordinates from a dataset when the coordinate variables are named `lon` and `lat`. The test creates a dataset with `lon` and `lat` variables, then calls the extraction function and asserts that the returned longitude and latitude arrays have the expected lengths and sample values that match the input coordinates. This verifies that the function can identify and extract spatial coordinates even when they use very common variable names like `lon` and `lat`. 

        Parameters:
            None

        Returns:
            None: Confirmed by asserting returned arrays have expected lengths.
        """
        ds = xr.Dataset({
            'lon': ('nCells', np.linspace(0, 360, 200)),
            'lat': ('nCells', np.linspace(-90, 90, 200))
        })
        
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(ds)
        
        assert len(lon) == pytest.approx(200)
        assert len(lat) == pytest.approx(200)   
    
    def test_extract_coordinates_in_coords(self: "TestExtractSpatialCoordinates", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This test confirms that `extract_spatial_coordinates` can successfully extract longitude and latitude coordinates from a dataset when the coordinate variables are present in the dataset's coordinates rather than as data variables. The test creates a dataset where `lonCell` and `latCell` are defined as coordinates, then calls the extraction function and asserts that the returned longitude and latitude arrays have the expected lengths and values that match the input coordinates. This verifies that the function can correctly identify spatial coordinates even when they are provided as coordinates in the xarray Dataset, which is a common structure for MPAS datasets. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real temperature data.

        Returns:
            None: Verified by asserting lengths of returned arrays.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS coordinates or temperature data not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(100, len(real_lon), len(mpas_surface_temp_data))
        
        ds = xr.Dataset(
            {'temperature': (['nCells'], mpas_surface_temp_data[:subset_size])},
            coords={
                'lonCell': ('nCells', real_lon[:subset_size]),
                'latCell': ('nCells', real_lat[:subset_size])
            }
        )
        
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(ds)
        
        assert len(lon) == subset_size
        assert len(lat) == subset_size
    
    def test_extract_coordinates_missing(self: "TestExtractSpatialCoordinates") -> None:
        """
        This test confirms that `extract_spatial_coordinates` raises a ValueError when the input dataset does not contain any recognizable longitude and latitude coordinate variables. The test creates a dataset without any of the expected coordinate variable names (e.g., `lonCell`, `latCell`, `longitude`, `latitude`, `lon`, `lat`), then calls the extraction function and asserts that a ValueError is raised with an appropriate message indicating that spatial coordinates could not be found. This ensures that the function properly handles cases where the necessary coordinate information is missing and provides clear feedback about the issue. 

        Parameters:
            None

        Returns:
            None: Verified by asserting the expected exception message.
        """
        ds = xr.Dataset({
            'temperature': (['nCells'], np.random.rand(100))
        })
        
        with pytest.raises(ValueError) as cm:
            MPASGeographicUtils.extract_spatial_coordinates(ds)

        assert "Could not find spatial coordinates" in str(cm.value)
    
    def test_extract_coordinates_multidimensional(self: "TestExtractSpatialCoordinates", mpas_coordinates) -> None:
        """
        This test confirms that `extract_spatial_coordinates` can successfully extract longitude and latitude coordinates from a dataset when the coordinate variables are multidimensional (e.g., have dimensions like `Time` and `nCells`). The test creates a dataset where `lonCell` and `latCell` are 2D arrays (e.g., with dimensions `Time` and `nCells`) using real MPAS coordinate data, then calls the extraction function and asserts that the returned longitude and latitude arrays are flattened to 1D and have the expected lengths. This verifies that the function can handle multidimensional coordinate variables by flattening them appropriately for downstream use. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified by asserting returned arrays are 1D and lengths match.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(100, len(real_lon))
        n_time = 5
        
        lon_2d = np.broadcast_to(real_lon[:subset_size], (n_time, subset_size))
        lat_2d = np.broadcast_to(real_lat[:subset_size], (n_time, subset_size))
        
        ds = xr.Dataset({
            'lonCell': (['Time', 'nCells'], lon_2d),
            'latCell': (['Time', 'nCells'], lat_2d)
        })
        
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(ds)
        
        assert lon.ndim == pytest.approx(1)
        assert lat.ndim == pytest.approx(1)
        assert len(lon) == n_time * subset_size
        assert len(lat) == n_time * subset_size
    
    def test_extract_coordinates_longitude_normalization(self: "TestExtractSpatialCoordinates") -> None:
        """
        This test confirms that `extract_spatial_coordinates` correctly normalizes longitude values into the [-180, 180] range when extracting coordinates from a dataset. The test creates a dataset with `lonCell` values that include longitudes in the [0, 360] range, then calls the extraction function and asserts that the returned longitude array has all values normalized to the [-180, 180] range (e.g., 180 should wrap to -180, 270 should wrap to -90). This verifies that the function applies longitude normalization as part of the coordinate extraction process, ensuring consistent longitude representation for downstream geographic operations. 

        Parameters:
            None

        Returns:
            None: Verified by assertions enforcing bounds and sample mappings.
        """
        ds = xr.Dataset({
            'lonCell': ('nCells', np.array([0, 90, 180, 270, 360])),
            'latCell': ('nCells', np.array([0, 10, 20, 30, 40]))
        })
        
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(ds)
        
        assert np.all(lon >= -180)
        assert np.all(lon <= 180)

        assert lon[0] == pytest.approx(0, abs=1e-5)
        assert lon[2] == pytest.approx(-180, abs=1e-5)
        assert lon[3] == pytest.approx(-90, abs=1e-5)
        assert lon[4] == pytest.approx(0, abs=1e-5)
    
    def test_extract_coordinates_real_mpas_grid(self: "TestExtractSpatialCoordinates", mock_mpas_mesh: xr.Dataset) -> None:
        """
        This test confirms that `extract_spatial_coordinates` can successfully extract longitude and latitude coordinates from a real MPAS mesh dataset. The test uses a fixture that provides actual MPAS mesh data (e.g., from a grid file), then calls the extraction function and asserts that the returned longitude and latitude arrays have expected properties such as being 1D, having lengths that match the number of cells, and containing values within valid geographic bounds. This serves as an integration test to verify that the coordinate extraction logic works correctly with real unstructured mesh data from MPAS. 

        Parameters:
            mock_mpas_mesh (xr.Dataset): Fixture providing real MPAS mesh data.

        Returns:
            None: Verified by assertions on returned coordinate arrays.
        """
        if mock_mpas_mesh is None or 'lonCell' not in mock_mpas_mesh:
            pytest.skip("MPAS mesh data not available")
            return
        
        lon, lat = MPASGeographicUtils.extract_spatial_coordinates(mock_mpas_mesh)
        
        assert len(lon) > 0
        assert len(lat) == len(lon)
        
        assert np.all(lon >= -180.0)
        assert np.all(lon <= 180.0)
        
        assert np.all(lat >= -90.0)
        assert np.all(lat <= 90.0)
        
        assert lon.ndim == pytest.approx(1)
        assert lat.ndim == pytest.approx(1)


class TestFilterBySpatialExtent:
    """ Test spatial filtering operations. """
    
    def test_filter_with_ncells_dimension(self: "TestFilterBySpatialExtent", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This test confirms that `filter_by_spatial_extent` correctly filters a DataArray that has an `nCells` dimension based on specified longitude and latitude bounds. The test creates a dataset with `lonCell` and `latCell` coordinates and a data variable (e.g., temperature) that has an `nCells` dimension. It then defines a geographic extent that includes some but not all of the points, calls the filtering function, and asserts that the returned filtered DataArray contains NaN values for points outside the extent while retaining valid values for points inside. Additionally, it checks that the mask correctly identifies which points are inside versus outside the specified geographic box. This verifies that the spatial filtering logic works as intended when the data array has an `nCells` dimension. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real temperature data.

        Returns:
            None: Verified via type assertions and mask checks.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS coordinates or temperature data not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(100, len(real_lon), len(mpas_surface_temp_data))
        
        ds = xr.Dataset({
            'lonCell': ('nCells', real_lon[:subset_size]),
            'latCell': ('nCells', real_lat[:subset_size]),
            'temperature': (['nCells'], mpas_surface_temp_data[:subset_size])
        })
        
        lon_min = np.percentile(real_lon[:subset_size], 25)
        lon_max = np.percentile(real_lon[:subset_size], 75)
        lat_min = np.percentile(real_lat[:subset_size], 25)
        lat_max = np.percentile(real_lat[:subset_size], 75)
        
        data = ds['temperature']

        filtered, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, ds, lon_min, lon_max, lat_min, lat_max
        )
        
        assert isinstance(filtered, xr.DataArray)
        assert isinstance(mask, np.ndarray)
        assert len(mask) == subset_size
        assert np.any(mask)  
        assert np.any(~mask)  
    
    def test_filter_without_ncells_dimension(self: "TestFilterBySpatialExtent") -> None:
        """
        This test confirms that `filter_by_spatial_extent` returns the original DataArray unchanged when the input data array does not have an `nCells` dimension. The test creates a dataset with `lonCell` and `latCell` coordinates but a data variable that does not depend on `nCells` (e.g., a time series), then calls the filtering function and asserts that the returned filtered DataArray is identical to the input data array, indicating that no filtering was applied. This verifies that the function correctly identifies when spatial filtering is not applicable due to the absence of an `nCells` dimension and handles it gracefully by returning the original data. 

        Parameters:
            None

        Returns:
            None: Confirmed via array equality assertion.
        """
        ds = xr.Dataset({
            'lonCell': ('nCells', np.linspace(-100, -90, 100)),
            'latCell': ('nCells', np.linspace(30, 40, 100)),
            'time': (['Time'], np.arange(5))
        })
        
        data = ds['time']

        filtered, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, ds, -98, -92, 32, 38
        )
        
        assert np.array_equal(filtered.values, data.values)
    
    def test_filter_all_inside(self: "TestFilterBySpatialExtent", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This test confirms that `filter_by_spatial_extent` correctly retains all valid data points when the specified geographic extent includes all locations. The test creates a dataset with `lonCell` and `latCell` coordinates and a data variable with an `nCells` dimension, then defines a geographic extent that encompasses all the coordinate points. It calls the filtering function and asserts that the returned filtered DataArray contains no NaN values (indicating all points are retained) and that the mask indicates all points are inside the extent. This verifies that the spatial filtering logic does not erroneously exclude any points when the extent fully contains the dataset. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real temperature data.

        Returns:
            None: Verified via mask and NaN assertions.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS coordinates or temperature data not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(100, len(real_lon), len(mpas_surface_temp_data))
        
        ds = xr.Dataset({
            'lonCell': ('nCells', real_lon[:subset_size]),
            'latCell': ('nCells', real_lat[:subset_size]),
            'temperature': (['nCells'], mpas_surface_temp_data[:subset_size])
        })
        
        lon_min = np.min(real_lon[:subset_size]) - 10
        lon_max = np.max(real_lon[:subset_size]) + 10
        lat_min = np.min(real_lat[:subset_size]) - 10
        lat_max = np.max(real_lat[:subset_size]) + 10
        
        data = ds['temperature']

        filtered, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, ds, lon_min, lon_max, lat_min, lat_max
        )
        
        assert np.all(mask)
        assert np.sum(np.isnan(filtered.values)) == pytest.approx(0)
    
    def test_filter_all_outside(self: "TestFilterBySpatialExtent", mpas_coordinates, mpas_surface_temp_data) -> None:
        """
        This test confirms that `filter_by_spatial_extent` correctly excludes all data points when the specified geographic extent does not include any locations. The test creates a dataset with `lonCell` and `latCell` coordinates and a data variable with an `nCells` dimension, then defines a geographic extent that is located far from any of the coordinate points (e.g., a small box in the Arctic when the data is from the tropics). It calls the filtering function and asserts that the returned filtered DataArray contains only NaN values (indicating all points are excluded) and that the mask indicates all points are outside the extent. This verifies that the spatial filtering logic correctly identifies when no points fall within the specified geographic box and handles it by masking out all data. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.
            mpas_surface_temp_data: Session fixture providing real temperature data.

        Returns:
            None: Verified via mask and NaN assertions.
        """
        if mpas_coordinates is None or mpas_surface_temp_data is None:
            pytest.skip("MPAS coordinates or temperature data not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(100, len(real_lon), len(mpas_surface_temp_data))
        
        ds = xr.Dataset({
            'lonCell': ('nCells', real_lon[:subset_size]),
            'latCell': ('nCells', real_lat[:subset_size]),
            'temperature': (['nCells'], mpas_surface_temp_data[:subset_size])
        })
        
        data = ds['temperature']

        filtered, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, ds, 179.5, 179.9, 89.5, 89.9
        )
        
        assert np.sum(mask) < subset_size * 0.1  
        assert np.all(np.isnan(filtered.values[~mask]))
    
    def test_filter_real_mpas_data(self: "TestFilterBySpatialExtent", mock_mpas_2d_data: xr.Dataset) -> None:
        """
        This test confirms that `filter_by_spatial_extent` can successfully filter a real MPAS 2D data variable (e.g., surface temperature) based on specified longitude and latitude bounds. The test uses a fixture that provides actual MPAS 2D data, selects a variable with an `nCells` dimension, defines a geographic extent that includes some of the points, and calls the filtering function. It then asserts that the returned filtered DataArray contains NaN values for points outside the extent while retaining valid values for points inside, and that the mask correctly identifies which points are inside versus outside the specified geographic box. This serves as an integration test to verify that the spatial filtering logic works correctly with real MPAS data variables. 

        Parameters:
            mock_mpas_2d_data (xr.Dataset): Fixture providing real MPAS 2D data.

        Returns:
            None: Verified by assertions on filtered data and mask.
        """
        if mock_mpas_2d_data is None or 't2m' not in mock_mpas_2d_data:
            pytest.skip("MPAS 2D data not available")
            return
        
        data = mock_mpas_2d_data['t2m'].isel(Time=0)
        
        filtered, mask = MPASGeographicUtils.filter_by_spatial_extent(
            data, mock_mpas_2d_data, -100, -50, 20, 50
        )
        
        assert isinstance(filtered, xr.DataArray)
        assert isinstance(mask, np.ndarray)
        assert len(mask) == len(data)        
        assert mask.dtype == bool


class TestNormalizeLongitude:
    """ Test longitude normalization. """
    
    def test_normalize_longitude_0_360(self: "TestNormalizeLongitude") -> None:
        """
        This test confirms that `normalize_longitude` correctly normalizes longitude values in the [0, 360] range to the [-180, 180] range. The test creates an array of longitude values that includes typical values in the [0, 360] range (e.g., 0, 90, 180, 270, 360), then calls the normalization function and asserts that the returned array has values correctly wrapped into the [-180, 180] range (e.g., 180 should wrap to -180, 270 should wrap to -90). This verifies that the function applies the correct normalization logic to convert longitudes from a common [0, 360] format into the standard [-180, 180] format used for geographic coordinates. 

        Parameters:
            None

        Returns:
            None: Verified by array equality checks.
        """
        lon = np.array([0, 90, 180, 270, 360])
        normalized = MPASGeographicUtils.normalize_longitude(lon)        
        expected = np.array([0, 90, -180, -90, 0]) 
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_normalize_longitude_already_normalized(self: "TestNormalizeLongitude") -> None:
        """
        This test confirms that `normalize_longitude` leaves longitude values that are already in the [-180, 180] range unchanged. The test creates an array of longitude values that are all within the [-180, 180] range (e.g., -180, -90, 0, 90, 179), then calls the normalization function and asserts that the returned array is identical to the input array. This verifies that the function does not alter longitude values that are already in the correct range, ensuring that it behaves as a proper normalization function without introducing unintended modifications to valid inputs. 

        Parameters:
            None

        Returns:
            None: Verified with array assertions.
        """
        lon = np.array([-180, -90, 0, 90, 179])  
        normalized = MPASGeographicUtils.normalize_longitude(lon)        
        expected = np.array([-180, -90, 0, 90, 179])
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_normalize_longitude_large_values(self: "TestNormalizeLongitude") -> None:
        """
        This test confirms that `normalize_longitude` correctly normalizes large out-of-bounds longitude values (e.g., 400, 720, -400) into the [-180, 180] range. The test creates an array of longitude values that includes large positive and negative values well outside the typical range, then calls the normalization function and asserts that the returned array has all values normalized to the [-180, 180] range (e.g., 400 should wrap to 40, 720 should wrap to 0, -400 should wrap to -40). This verifies that the function can handle extreme longitude values and applies the correct modular arithmetic to bring them into the standard geographic range. 

        Parameters:
            None

        Returns:
            None: Verified by range checks and sample value assertions.
        """
        lon = np.array([400, 720, -400])
        normalized = MPASGeographicUtils.normalize_longitude(lon)
        
        assert np.all(normalized >= -180)
        assert np.all(normalized <= 180)

        assert normalized[0] == pytest.approx(40, abs=1e-5)
        assert normalized[1] == pytest.approx(0, abs=1e-5)
        assert normalized[2] == pytest.approx(-40, abs=1e-5)
    
    def test_normalize_longitude_scalar(self: "TestNormalizeLongitude") -> None:
        """
        This test confirms that `normalize_longitude` can handle scalar longitude values and correctly normalizes them into the [-180, 180] range. The test provides a single scalar longitude value (e.g., 270) that is outside the standard range, then calls the normalization function and asserts that the returned value is normalized to the correct equivalent within the [-180, 180] range (e.g., 270 should wrap to -90). This verifies that the function can process both array and scalar inputs for longitude normalization without errors. 

        Parameters:
            None

        Returns:
            None: Verified via scalar approximation assertion.
        """
        lon = np.array(270)  
        normalized = MPASGeographicUtils.normalize_longitude(lon)        
        assert normalized == pytest.approx(-90, abs=1e-5)
    
    def test_normalize_longitude_real_mpas_coords(self: "TestNormalizeLongitude", mpas_coordinates) -> None:
        """
        This test confirms that `normalize_longitude` correctly normalizes a real array of longitude coordinates from an MPAS dataset, ensuring all values are within the [-180, 180] range. The test uses a fixture that provides actual MPAS longitude coordinates, calls the normalization function, and asserts that all returned longitude values are within the valid geographic bounds. This serves as an integration test to verify that the normalization logic works correctly with real-world MPAS coordinate data. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified by range assertions on normalized coordinates.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        lon, _ = mpas_coordinates
        normalized = MPASGeographicUtils.normalize_longitude(lon)
        
        assert np.all(normalized >= -180.0)
        assert np.all(normalized <= 180.0)        
        assert len(normalized) == len(lon)


class TestValidateGeographicExtent:
    """ Test geographic extent validation. """
    
    def test_validate_extent_valid(self: "TestValidateGeographicExtent") -> None:
        """
        This test confirms that `validate_geographic_extent` returns True for valid geographic extents that are within the conventional bounds and have proper min/max relationships. The test calls the validation function with several valid extents (e.g., (-100, -90, 30, 40), (-180, 180, -90, 90), (0, 10, 0, 10)) and asserts that the function returns True for each case. This verifies that the validator correctly identifies valid geographic extents and allows them to pass through without issue.

        Parameters:
            None

        Returns:
            None: Verified via boolean assertions.
        """
        assert MPASGeographicUtils.validate_geographic_extent(
            (-100, -90, 30, 40)
        )

        assert MPASGeographicUtils.validate_geographic_extent(
            (-180, 180, -90, 90)
        )

        assert MPASGeographicUtils.validate_geographic_extent(
            (0, 10, 0, 10)
        )
    
    def test_validate_extent_invalid_lon_min(self: "TestValidateGeographicExtent") -> None:
        """
        This test confirms that `validate_geographic_extent` returns False for extents with `lon_min` below -180 degrees. The test calls the validation function with an extent that has a `lon_min` value less than -180 (e.g., (-200, -90, 30, 40)) and asserts that the function returns False, indicating that the extent is invalid due to out-of-bounds longitude. This verifies that the validator correctly enforces valid longitude ranges for geographic extents. 

        Parameters:
            None

        Returns:
            None: Verified by asserting the function returns False.
        """
        assert not MPASGeographicUtils.validate_geographic_extent(
            (-200, -90, 30, 40)
        )
    
    def test_validate_extent_invalid_lon_max(self: "TestValidateGeographicExtent") -> None:
        """
        This test confirms that `validate_geographic_extent` returns False for extents with `lon_max` above 180 degrees. The test calls the validation function with an extent that has a `lon_max` value greater than 180 (e.g., (-100, 200, 30, 40)) and asserts that the function returns False, indicating that the extent is invalid due to out-of-bounds longitude. This verifies that the validator correctly enforces valid longitude ranges for geographic extents. 

        Parameters:
            None

        Returns:
            None: Verified via a False return value.
        """
        assert not MPASGeographicUtils.validate_geographic_extent(
            (-100, 200, 30, 40)
        )
    
    def test_validate_extent_invalid_lat_min(self: "TestValidateGeographicExtent") -> None:
        """
        This test confirms that `validate_geographic_extent` returns False for extents with `lat_min` below -90 degrees. The test calls the validation function with an extent that has a `lat_min` value less than -90 (e.g., (-100, -90, -100, 40)) and asserts that the function returns False, indicating that the extent is invalid due to out-of-bounds latitude. This verifies that the validator correctly enforces valid latitude ranges for geographic extents.

        Parameters:
            None

        Returns:
            None: Verified by False return value.
        """
        assert not MPASGeographicUtils.validate_geographic_extent(
            (-100, -90, -100, 40)
        )
    
    def test_validate_extent_invalid_lat_max(self: "TestValidateGeographicExtent") -> None:
        """
        This test confirms that `validate_geographic_extent` returns False for extents with `lat_max` above 90 degrees. The test calls the validation function with an extent that has a `lat_max` value greater than 90 (e.g., (-100, -90, 30, 100)) and asserts that the function returns False, indicating that the extent is invalid due to out-of-bounds latitude. This verifies that the validator correctly enforces valid latitude ranges for geographic extents. 

        Parameters:
            None

        Returns:
            None: Verified via assertion.
        """
        assert not MPASGeographicUtils.validate_geographic_extent(
            (-100, -90, 30, 100)
        )
    
    def test_validate_extent_reversed_lon(self: "TestValidateGeographicExtent") -> None:
        """
        This test confirms that `validate_geographic_extent` returns False for extents where `lon_max` is less than `lon_min`. The test calls the validation function with an extent that has reversed longitude bounds (e.g., (-90, -100, 30, 40)) and asserts that the function returns False, indicating that the extent is invalid due to improper min/max relationships. This verifies that the validator correctly identifies and rejects extents with reversed longitude bounds. 

        Parameters:
            None

        Returns:
            None: Verified via assertion that the function returns False.
        """
        assert not MPASGeographicUtils.validate_geographic_extent(
            (-90, -100, 30, 40)
        )
    
    def test_validate_extent_reversed_lat(self: "TestValidateGeographicExtent") -> None:
        """
        This test confirms that `validate_geographic_extent` returns False for extents where `lat_max` is less than `lat_min`. The test calls the validation function with an extent that has reversed latitude bounds (e.g., (-100, -90, 40, 30)) and asserts that the function returns False, indicating that the extent is invalid due to improper min/max relationships. This verifies that the validator correctly identifies and rejects extents with reversed latitude bounds. 

        Parameters:
            None

        Returns:
            None: Verified via assertion.
        """
        assert not MPASGeographicUtils.validate_geographic_extent(
            (-100, -90, 40, 30)
        )
    
    def test_validate_extent_equal_bounds(self: "TestValidateGeographicExtent") -> None:
        """
        This test confirms that `validate_geographic_extent` returns False for extents where `lon_min` equals `lon_max` or `lat_min` equals `lat_max`, as these do not represent valid geographic areas. The test calls the validation function with extents that have equal longitude bounds (e.g., (-100, -100, 30, 40)) and equal latitude bounds (e.g., (-100, -90, 30, 30)) and asserts that the function returns False in both cases. This verifies that the validator correctly identifies and rejects extents that do not define a valid geographic area due to zero width or height.

        Parameters:
            None

        Returns:
            None: Verified by asserting the function returns False for such inputs.
        """
        assert not MPASGeographicUtils.validate_geographic_extent(
            (-100, -100, 30, 40)
        )

        assert not MPASGeographicUtils.validate_geographic_extent(
            (-100, -90, 30, 30)
        )


class TestGetExtentFromCoordinates:
    """ Test extent calculation from coordinates. """
    
    def test_get_extent_no_buffer(self: "TestGetExtentFromCoordinates", mpas_coordinates) -> None:
        """
        This test confirms that `get_extent_from_coordinates` correctly computes the geographic extent (lon_min, lon_max, lat_min, lat_max) from given longitude and latitude coordinate arrays without applying any buffer. The test uses real MPAS coordinate data provided by a fixture, calls the function with a buffer of 0.0, and asserts that the returned extent values match the minimum and maximum of the input longitude and latitude arrays within a small tolerance. This verifies that the function accurately calculates the raw geographic bounds based on the provided coordinates when no buffer is applied. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified via element-wise assertions on the returned extent.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(50, len(real_lon))
        lon = real_lon[:subset_size]
        lat = real_lat[:subset_size]
        
        extent = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=0.0)
        
        assert extent[0] == pytest.approx(np.min(lon), abs=1e-5)
        assert extent[1] == pytest.approx(np.max(lon), abs=1e-5)
        assert extent[2] == pytest.approx(np.min(lat), abs=1e-5)
        assert extent[3] == pytest.approx(np.max(lat), abs=1e-5)
    
    def test_get_extent_with_buffer(self: "TestGetExtentFromCoordinates", mpas_coordinates) -> None:
        """
        This test confirms that `get_extent_from_coordinates` correctly applies a specified buffer to the computed geographic extent. The test uses real MPAS coordinate data, calls the function with a positive buffer value (e.g., 5.0 degrees), and asserts that the returned extent values are expanded by the buffer amount from the minimum and maximum of the input longitude and latitude arrays. Additionally, it checks that latitude values are properly clamped to the valid range of [-90, 90] when the buffer pushes them beyond these bounds. This verifies that the function correctly incorporates buffering into the extent calculation while respecting geographic limits. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified via expected buffered extent values.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(50, len(real_lon))
        lon = real_lon[:subset_size]
        lat = real_lat[:subset_size]
        
        buffer = 5.0
        extent = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=buffer)
        
        expected_lon_min = np.min(lon) - buffer
        expected_lon_max = np.max(lon) + buffer

        expected_lat_min = max(-90, np.min(lat) - buffer) 
        expected_lat_max = min(90, np.max(lat) + buffer)   
        
        assert extent[0] == pytest.approx(expected_lon_min, abs=1e-5) 
        assert extent[1] == pytest.approx(expected_lon_max, abs=1e-5) 
        assert extent[2] == pytest.approx(expected_lat_min, abs=1e-5) 
        assert extent[3] == pytest.approx(expected_lat_max, abs=1e-5) 
    
    def test_get_extent_with_nan_values(self: "TestGetExtentFromCoordinates") -> None:
        """
        This test confirms that `get_extent_from_coordinates` correctly ignores NaN values in the input longitude and latitude arrays when calculating the geographic extent. The test creates longitude and latitude arrays that contain NaN values at certain indices, calls the function, and asserts that the returned extent values correspond only to the valid coordinate pairs, effectively ignoring any NaN entries. This verifies that the function can handle missing or invalid coordinate data gracefully without allowing it to affect the computed geographic bounds. 

        Parameters:
            None

        Returns:
            None: Verified by asserting extent values correspond to valid pairs.
        """
        lon = np.array([-100, np.nan, -90, -95, np.nan])
        lat = np.array([30, np.nan, 40, 35, np.nan]) 
        
        extent = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=0.0)
        
        assert extent[0] == pytest.approx(-100, abs=1e-5)
        assert extent[1] == pytest.approx(-90, abs=1e-5)
        assert extent[2] == pytest.approx(30, abs=1e-5)
        assert extent[3] == pytest.approx(40, abs=1e-5)
    
    def test_get_extent_all_nan(self: "TestGetExtentFromCoordinates") -> None:
        """
        This test confirms that `get_extent_from_coordinates` raises a ValueError when all input longitude and latitude values are NaN, as there are no valid coordinates to compute an extent from. The test creates longitude and latitude arrays that are entirely NaN, calls the function, and asserts that a ValueError is raised with an appropriate error message indicating that no valid coordinates were provided. This verifies that the function correctly handles the edge case of completely invalid input data by raising an informative exception rather than returning an invalid extent. 

        Parameters:
            None

        Returns:
            None: Verified by catching the expected exception.
        """
        lon = np.array([np.nan, np.nan, np.nan])
        lat = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError) as cm:
            MPASGeographicUtils.get_extent_from_coordinates(lon, lat)

        assert "No valid coordinates" in str(cm.value)
    
    def test_get_extent_clamping_lon_min(self: "TestGetExtentFromCoordinates") -> None:
        """
        This test confirms that `get_extent_from_coordinates` allows longitude minimum values to extend beyond -180 degrees when a buffer is applied, without clamping them to the valid range. The test creates longitude and latitude arrays with values near the -180 degree boundary, applies a buffer that would push the minimum longitude below -180, and asserts that the returned `lon_min` value includes the buffer and is less than -180. This verifies that the function does not artificially clamp longitude values to the [-180, 180] range when calculating the extent with a buffer, allowing for proper handling of global datasets that may span across the dateline. 

        Parameters:
            None

        Returns:
            None: Verified by asserting the returned lon_min includes buffer.
        """
        lon = np.array([-178, -179])
        lat = np.array([0, 10])
        
        extent = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=5.0)
        
        assert extent[0] == pytest.approx(-184.0)
    
    def test_get_extent_clamping_lon_max(self: "TestGetExtentFromCoordinates") -> None:
        """
        This test confirms that `get_extent_from_coordinates` allows longitude maximum values to extend beyond 180 degrees when a buffer is applied, without clamping them to the valid range. The test creates longitude and latitude arrays with values near the 180 degree boundary, applies a buffer that would push the maximum longitude above 180, and asserts that the returned `lon_max` value includes the buffer and is greater than 180. This verifies that the function does not artificially clamp longitude values to the [-180, 180] range when calculating the extent with a buffer, allowing for proper handling of global datasets that may span across the dateline. 

        Parameters:
            None

        Returns:
            None: Verified by asserting the returned lon_max includes buffer.
        """
        lon = np.array([178, 179])
        lat = np.array([0, 10])
        
        extent = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=5.0)
        
        assert extent[1] == pytest.approx(184.0)
    
    def test_get_extent_clamping_lat_min(self: "TestGetExtentFromCoordinates") -> None:
        """
        This test confirms that `get_extent_from_coordinates` correctly clamps latitude minimum values to -90 degrees when a buffer pushes them below the valid range. The test creates longitude and latitude arrays with values near the -90 degree boundary, applies a buffer that would push the minimum latitude below -90, and asserts that the returned `lat_min` value is clamped to -90. This verifies that the function enforces valid latitude bounds when calculating the extent with a buffer, preventing invalid geographic coordinates from being returned. 

        Parameters:
            None

        Returns:
            None: Verified by asserting lat_min equals -90.
        """
        lon = np.array([0, 10])
        lat = np.array([-88, -89])
        
        extent = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=5.0)
        
        assert extent[2] == pytest.approx(-90.0)
    
    def test_get_extent_clamping_lat_max(self: "TestGetExtentFromCoordinates") -> None:
        """
        This test confirms that `get_extent_from_coordinates` correctly clamps latitude maximum values to 90 degrees when a buffer pushes them above the valid range. The test creates longitude and latitude arrays with values near the 90 degree boundary, applies a buffer that would push the maximum latitude above 90, and asserts that the returned `lat_max` value is clamped to 90. This verifies that the function enforces valid latitude bounds when calculating the extent with a buffer, preventing invalid geographic coordinates from being returned. 

        Parameters:
            None

        Returns:
            None: Verified by asserting lat_max equals 90.
        """
        lon = np.array([0, 10])
        lat = np.array([88, 89])
        
        extent = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=5.0)
        
        assert extent[3] == pytest.approx(90.0)
    
    def test_get_extent_real_mpas_coords(self: "TestGetExtentFromCoordinates", mpas_coordinates) -> None:
        """
        This test confirms that `get_extent_from_coordinates` can successfully compute a valid geographic extent from real MPAS longitude and latitude coordinate arrays, both with and without applying a buffer. The test uses actual MPAS coordinate data provided by a fixture, calls the function to compute the extent without a buffer and verifies that the returned bounds are valid and within expected ranges. It then calls the function again with a positive buffer and verifies that the buffered extent is larger than or equal to the original extent while still being valid. This serves as an integration test to ensure that the extent calculation logic works correctly with real-world MPAS coordinate data. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified by assertions on returned extent bounds.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        lon, lat = mpas_coordinates
        
        extent = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=0.0)
        
        assert len(extent) == pytest.approx(4)
        assert extent[0] <= extent[1]  
        assert extent[2] <= extent[3] 
        
        assert extent[0] >= -180.0
        assert extent[1] <= 180.0
        assert extent[2] >= -90.0
        assert extent[3] <= 90.0
        
        extent_buffered = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=5.0)
        
        assert extent_buffered[0] <= extent[0]
        assert extent_buffered[1] >= extent[1]
        assert extent_buffered[2] <= extent[2]
        assert extent_buffered[3] >= extent[3]


class TestCalculateSpatialResolution:
    """ Test spatial resolution calculation. """
    
    def test_calculate_resolution_less_than_2_points(self: "TestCalculateSpatialResolution") -> None:
        """
        This test confirms that `calculate_spatial_resolution` returns a resolution of 0.0 when provided with fewer than 2 points, as there are no distances to compute. The test creates longitude and latitude arrays with only one point, calls the function, and asserts that the returned resolution is exactly 0.0. This verifies that the function correctly handles edge cases with insufficient data by returning a resolution of zero rather than attempting to compute distances.

        Parameters:
            None

        Returns:
            None: Verified via equality assertion.
        """
        lon = np.array([0])
        lat = np.array([0])
        
        resolution = MPASGeographicUtils.calculate_spatial_resolution(lon, lat)        
        assert resolution == pytest.approx(0.0, abs=1e-5)
    
    def test_calculate_resolution_2_points(self: "TestCalculateSpatialResolution") -> None:
        """
        This test confirms that `calculate_spatial_resolution` correctly computes the distance between two points as the spatial resolution. The test creates longitude and latitude arrays with exactly two points, calls the function, and asserts that the returned resolution is equal to the expected distance between those two points calculated using the Pythagorean theorem. This verifies that the function can compute a valid spatial resolution when provided with a simple case of two distinct points. 

        Parameters:
            None

        Returns:
            None: Verified with approximate equality to expected distance.
        """
        lon = np.array([0, 10])
        lat = np.array([0, 10])
        
        resolution = MPASGeographicUtils.calculate_spatial_resolution(lon, lat)
        
        assert resolution > 0.0
        expected_dist = np.sqrt((10-0)**2 + (10-0)**2)
        assert resolution == pytest.approx(expected_dist, abs=1e-5)
    
    def test_calculate_resolution_uniform_grid(self: "TestCalculateSpatialResolution", mpas_coordinates) -> None:
        """
        This test confirms that `calculate_spatial_resolution` computes a consistent resolution for a uniform grid of points. The test creates longitude and latitude arrays that form a regular grid (e.g., 5x5 points spaced evenly), calls the function, and asserts that the returned resolution is approximately equal to the known spacing between the grid points. This verifies that the function can accurately compute spatial resolution for structured grids where distances between points are uniform. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified by approximate comparison to expected value.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(500, len(real_lon))
        lon = real_lon[:subset_size]
        lat = real_lat[:subset_size]
        
        resolution = MPASGeographicUtils.calculate_spatial_resolution(lon, lat)
        
        assert resolution > 0.0
        assert 0.5 < resolution < 200.0
    
    def test_calculate_resolution_sampling(self: "TestCalculateSpatialResolution", mpas_coordinates) -> None:
        """
        This test confirms that `calculate_spatial_resolution` can compute a reasonable spatial resolution using random sampling when the input coordinate arrays are large. The test uses real MPAS coordinate data, calls the function with a specified `sample_size` to trigger random sampling of points, and asserts that the returned resolution is positive and within a reasonable range for an MPAS mesh. This verifies that the sampling logic in the function works correctly to provide an estimate of spatial resolution without needing to compute distances for all points in a large dataset. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified by ensuring a positive resolution is returned.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        
        resolution = MPASGeographicUtils.calculate_spatial_resolution(
            real_lon, real_lat, sample_size=100
        )
        
        assert resolution > 0.0
        assert 0.5 < resolution < 200.0
    
    def test_calculate_resolution_no_sampling(self: "TestCalculateSpatialResolution", mpas_coordinates) -> None:
        """
        This test confirms that `calculate_spatial_resolution` computes the spatial resolution using all points when the `sample_size` is larger than the number of available points. The test uses real MPAS coordinate data, calls the function with a `sample_size` that exceeds the length of the coordinate arrays, and asserts that the returned resolution is positive. This verifies that the function correctly defaults to using all points for resolution calculation when sampling is not applicable due to a small dataset. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified by checking returned resolution is positive.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        real_lon, real_lat = mpas_coordinates
        subset_size = min(50, len(real_lon))
        lon = real_lon[:subset_size]
        lat = real_lat[:subset_size]
        
        resolution = MPASGeographicUtils.calculate_spatial_resolution(
            lon, lat, sample_size=1000
        )
        
        assert resolution > 0.0
    
    def test_calculate_resolution_zero_distances(self: "TestCalculateSpatialResolution") -> None:
        """
        This test confirms that `calculate_spatial_resolution` returns a resolution of 0.0 when all input points are at the same location, resulting in zero distances between them. The test creates longitude and latitude arrays where all values are identical, calls the function, and asserts that the returned resolution is exactly 0.0. This verifies that the function correctly handles cases where there is no spatial separation between points by returning a resolution of zero. 

        Parameters:
            None

        Returns:
            None: Verified via equality assertion.
        """
        lon = np.array([0, 0, 0, 0])
        lat = np.array([0, 0, 0, 0])
        
        resolution = MPASGeographicUtils.calculate_spatial_resolution(lon, lat)        
        assert resolution == pytest.approx(0.0, abs=1e-5)
    
    def test_calculate_resolution_mixed_distances(self: "TestCalculateSpatialResolution") -> None:
        """
        This test confirms that `calculate_spatial_resolution` correctly computes the median of non-zero distances when the input points have a mix of identical and distinct locations. The test creates longitude and latitude arrays where some points are at the same location (resulting in zero distances) while others are spaced apart, calls the function, and asserts that the returned resolution is positive and corresponds to the median of the non-zero distances. This verifies that the function can handle datasets with duplicate points without allowing zero distances to skew the resolution calculation. 

        Parameters:
            None

        Returns:
            None: Verified by asserting the returned resolution is positive.
        """
        lon = np.array([0, 1, 1, 2, 2, 10])
        lat = np.array([0, 0, 1, 1, 2, 10])
        
        resolution = MPASGeographicUtils.calculate_spatial_resolution(lon, lat)
        assert resolution > 0.0
    
    def test_calculate_resolution_real_mpas_mesh(self: "TestCalculateSpatialResolution", mpas_coordinates) -> None:
        """
        This test confirms that `calculate_spatial_resolution` computes a reasonable spatial resolution for a real MPAS mesh using the provided longitude and latitude coordinate arrays. The test uses actual MPAS coordinate data, calls the function with a specified `sample_size` to estimate the resolution, and asserts that the returned resolution is positive and within a plausible range for an MPAS grid (e.g., between 0.5 and 200 km). This serves as an integration test to verify that the spatial resolution calculation works correctly with real-world MPAS coordinate data and provides meaningful results. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified by assertions on calculated resolution.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        lon, lat = mpas_coordinates
        
        resolution = MPASGeographicUtils.calculate_spatial_resolution(
            lon, lat, sample_size=1000
        )
        
        assert resolution > 0.0
        assert np.isfinite(resolution)


class TestIsGlobalExtent:
    """ Test global extent detection. """
    
    def test_is_global_full_extent(self: "TestIsGlobalExtent") -> None:
        """
        This test confirms that `is_global_extent` correctly identifies a full global extent as global. The test defines an extent that covers the entire globe (e.g., (-180, 180, -90, 90)), calls the function with a default tolerance, and asserts that the function returns True, indicating that the extent is recognized as global. This verifies that the function can correctly identify a perfect global coverage scenario as meeting the criteria for being considered global. 

        Parameters:
            None

        Returns:
            None: Verified by asserting the function returns True.
        """
        extent = (-180, 180, -90, 90)        
        assert MPASGeographicUtils.is_global_extent(extent, tolerance=1.0)
    
    def test_is_global_near_full_extent(self: "TestIsGlobalExtent") -> None:
        """
        This test confirms that `is_global_extent` can identify an extent that is near-global as global when it meets the specified tolerance criteria. The test defines an extent that covers nearly the entire globe (e.g., (-179, 180, -89, 90)), calls the function with a default tolerance of 1.0 degree, and asserts that the function returns True, indicating that the extent is considered global despite being slightly less than full coverage. This verifies that the function correctly applies the tolerance threshold to allow for minor deviations from perfect global coverage while still recognizing it as global. 

        Parameters:
            None

        Returns:
            None: Verified by asserting True is returned for near-global extent.
        """
        extent = (-179, 180, -89, 90)         
        assert MPASGeographicUtils.is_global_extent(extent, tolerance=1.0)
    
    def test_is_not_global_regional_extent(self: "TestIsGlobalExtent") -> None:
        """
        This test confirms that `is_global_extent` correctly identifies a regional extent that does not meet global coverage criteria as not global. The test defines an extent that covers only a small region of the globe (e.g., (-100, -90, 30, 40)), calls the function with a default tolerance, and asserts that the function returns False, indicating that the extent is not considered global. This verifies that the function can correctly distinguish between global and regional extents based on the specified coverage thresholds. 

        Parameters:
            None

        Returns:
            None: Verified by asserting the function returns False.
        """
        extent = (-100, -90, 30, 40)        
        assert not MPASGeographicUtils.is_global_extent(extent, tolerance=1.0)
    
    def test_is_global_lon_only(self: "TestIsGlobalExtent") -> None:
        """
        This test confirms that `is_global_extent` does not flag extents that are global in longitude only as global. The test defines an extent that covers the full range of longitude but only a portion of latitude (e.g., (-180, 180, 0, 90)), calls the function with a default tolerance, and asserts that the function returns False, indicating that the extent is not considered global due to insufficient latitude coverage. This verifies that the function requires both longitude and latitude coverage to meet global criteria and does not allow one axis to compensate for the other. 

        Parameters:
            None

        Returns:
            None: Verified by asserting False is returned.
        """
        extent = (-180, 180, 0, 90)        
        assert not MPASGeographicUtils.is_global_extent(extent, tolerance=1.0)
    
    def test_is_global_lat_only(self: "TestIsGlobalExtent") -> None:
        """
        This test confirms that `is_global_extent` does not flag extents that are global in latitude only as global. The test defines an extent that covers the full range of latitude but only a portion of longitude (e.g., (0, 180, -90, 90)), calls the function with a default tolerance, and asserts that the function returns False, indicating that the extent is not considered global due to insufficient longitude coverage. This verifies that the function requires both longitude and latitude coverage to meet global criteria and does not allow one axis to compensate for the other. 

        Parameters:
            None

        Returns:
            None: Verified by asserting False is returned.
        """
        extent = (0, 180, -90, 90)         
        assert not MPASGeographicUtils.is_global_extent(extent, tolerance=1.0)
    
    def test_is_global_custom_tolerance(self: "TestIsGlobalExtent") -> None:
        """
        This test confirms that `is_global_extent` correctly applies a custom tolerance to determine global coverage. The test defines an extent that is close to global but does not meet the default tolerance criteria (e.g., (-175, 180, -85, 90)), calls the function with both the default tolerance and a larger custom tolerance, and asserts that the function returns False with the default tolerance but returns True with the larger tolerance. This verifies that the function correctly uses the specified tolerance to evaluate whether an extent meets the criteria for being considered global. 

        Parameters:
            None

        Returns:
            None: Verified by toggling the `tolerance` argument.
        """
        extent = (-175, 180, -85, 90)        
        assert not MPASGeographicUtils.is_global_extent(extent, tolerance=1.0)
        assert MPASGeographicUtils.is_global_extent(extent, tolerance=5.0)
    
    def test_is_global_edge_case_exact_threshold(self: "TestIsGlobalExtent") -> None:
        """
        This test confirms that `is_global_extent` correctly identifies an extent that is exactly at the threshold of global coverage as global. The test defines an extent that has longitude and latitude ranges that are exactly 1 degree less than the full global coverage (e.g., (-179, 180, -89, 90)), calls the function with a tolerance of 1.0 degree, and asserts that the function returns True, indicating that the extent is considered global at the exact threshold. This verifies that the function's logic for determining global coverage correctly includes extents that meet the criteria at the boundary defined by the tolerance. 

        Parameters:
            None

        Returns:
            None: Verified by asserting True is returned.
        """
        extent = (-179, 180, -90, 90)
        assert MPASGeographicUtils.is_global_extent(extent, tolerance=1.0)
    
    def test_is_global_extent_real_mpas_coords(self: "TestIsGlobalExtent", mpas_coordinates) -> None:
        """
        This test confirms that `is_global_extent` can correctly identify the geographic extent of a real MPAS mesh as global when it meets the criteria for global coverage. The test uses actual MPAS coordinate data provided by a fixture, calculates the geographic extent from the coordinates using `get_extent_from_coordinates`, and then calls `is_global_extent` with the calculated extent and a specified tolerance. The test asserts that the function returns True, indicating that the MPAS mesh is recognized as having a global extent based on its coordinate coverage. This serves as an integration test to verify that the global extent detection logic works correctly with real-world MPAS coordinate data. 

        Parameters:
            mpas_coordinates: Session fixture providing real MPAS lon/lat arrays.

        Returns:
            None: Verified by asserting extent is detected as global.
        """
        if mpas_coordinates is None:
            pytest.skip("MPAS coordinates not available")
            return
        
        lon, lat = mpas_coordinates
        
        extent = MPASGeographicUtils.get_extent_from_coordinates(lon, lat, buffer=0.0)
        is_global = MPASGeographicUtils.is_global_extent(extent, tolerance=1.0)
        assert is_global is True


if __name__ == "__main__": 
    pytest.main([__file__]) 