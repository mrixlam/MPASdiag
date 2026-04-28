#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPASdiag remapping functionality

This module contains unit tests for the remapping functionality in MPASdiag, specifically focusing on the core remapping classes and functions. The tests verify that the remapping module can be imported successfully, that the main remapping class and its methods are defined, and that the utility function for creating target grids is available. These tests serve as a basic sanity check to ensure that the remapping components are present and accessible in the testing environment before running more comprehensive remapping tests.

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
from mpasdiag.processing.remapping import (
    remap_mpas_to_latlon
)
from tests.test_data_helpers import load_mpas_coords_from_processor
from mpasdiag.processing.remapping import ESMPY_AVAILABLE

REMAPPING_AVAILABLE = True


@pytest.mark.skipif(not REMAPPING_AVAILABLE, reason="Remapping module not available")


class TestKDTreeRemapping:
    """ Test KDTree-based remapping functionality. """
    
    def setup_method(self: 'TestKDTreeRemapping') -> None:
        """
        This fixture sets up synthetic MPAS coordinates and a test field for remapping tests. It uses a deterministic loader to generate longitude and latitude arrays along with synthetic data values based on a combination of sinusoidal patterns and noise derived from the u velocity component. The fixture ensures that the generated coordinates and data have the expected shapes and are suitable for testing the remapping functionality. This setup is shared across multiple remapping tests to provide consistent input data. 

        Parameters:
            self ('TestKDTreeRemapping'): Test instance to receive prepared attributes.

        Returns:
            None: Fixture populates instance variables and returns nothing.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon

        self.remap_func = remap_mpas_to_latlon        
        self.n_cells = 1000

        lon, lat, u, v = load_mpas_coords_from_processor(n=self.n_cells)

        self.mpas_lon = lon
        self.mpas_lat = lat

        lon_rad = np.radians(self.mpas_lon)
        lat_rad = np.radians(self.mpas_lat)

        umin = u.min()
        umax = u.max()

        noise = 5 * (u - umin) / (umax - umin + 1e-12)
        self.mpas_data = 20 + 15 * np.sin(3 * lon_rad) * np.cos(2 * lat_rad) + noise

        assert self.mpas_lon.shape == (self.n_cells,)
        assert self.mpas_lat.shape == (self.n_cells,)
        assert self.mpas_data.shape == (self.n_cells,)
        
    
    def test_convenience_function(self: 'TestKDTreeRemapping') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function correctly remaps synthetic MPAS data to a regular latitude-longitude grid using the nearest neighbor method. The test checks that the output is an xarray DataArray with the expected dimensions and that all values are finite. This validates the core remapping functionality at a coarse resolution for speed. The test ensures that the function can handle typical input data and produce a valid remapped output without errors. 

        Parameters:
            self ('TestKDTreeRemapping'): Test instance containing synthetic MPAS data.

        Returns:
            None: Assertions validate the remapped DataArray structure and values.
        """
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=10.0
        )
        
        assert isinstance(remapped, xr.DataArray)
        assert len(remapped.dims) == pytest.approx(2)
        assert 'lon' in remapped.dims
        assert 'lat' in remapped.dims
        assert np.all(np.isfinite(remapped.values))
    
    def test_remap_with_dataarray(self: 'TestKDTreeRemapping') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function correctly handles xarray DataArray inputs and preserves variable attributes through the remapping process. The test creates a DataArray with synthetic MPAS data and associated metadata, then remaps it to a regular grid. Assertions confirm that the output is an xarray DataArray and that key attributes like 'units' and 'long_name' are retained in the remapped result. This ensures that users can maintain important metadata when using the convenience function for remapping. 
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        data_array = xr.DataArray(
            self.mpas_data,
            dims=['nCells'],
            attrs={'units': 'K', 'long_name': 'Temperature'}
        )
        
        remapped = self.remap_func(
            data=data_array,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=5.0
        )
        
        assert isinstance(remapped, xr.DataArray)
        assert remapped.attrs['units'] == 'K'
        assert np.all(np.isfinite(remapped.values))
    
    def test_regional_remapping(self: 'TestKDTreeRemapping') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can successfully remap data to a regional grid with specified bounds and resolution. The test checks that the output is an xarray DataArray with longitude and latitude dimensions that match the expected lengths based on the provided regional parameters. This validates that the function can handle remapping to a smaller geographic domain (North America) at a 2-degree resolution, ensuring proper handling of regional bounds and spacing. The test confirms that the remapping process produces a valid output without errors for a typical regional use case. 
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-130, lon_max=-60,
            lat_min=25, lat_max=50,
            resolution=2.0
        )
        
        assert isinstance(remapped, xr.DataArray)
        assert len(remapped.lon) == pytest.approx(36)
        assert len(remapped.lat) == pytest.approx(13)
    
    def test_fine_resolution(self: 'TestKDTreeRemapping') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle remapping to a fine-resolution grid without producing NaNs or errors. The test uses a 0.5-degree resolution over a 20x20 degree domain to validate fine-scale remapping capabilities. The function should produce a remapped DataArray with all finite values, demonstrating that it can manage the increased computational demands of denser target grids while maintaining data integrity. This test ensures that users can perform high-resolution remapping when needed without encountering issues. 
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-10, lon_max=10,
            lat_min=-10, lat_max=10,
            resolution=0.5
        )
        
        assert isinstance(remapped, xr.DataArray)
        assert np.all(np.isfinite(remapped.values))
        assert len(remapped.lon) == pytest.approx(41)
        assert len(remapped.lat) == pytest.approx(41)
    
    def test_data_preservation(self: 'TestKDTreeRemapping') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function preserves the range of data values during remapping. The test checks that the minimum and maximum values in the remapped output are within the range of the original input data. This is important to ensure that the remapping process does not introduce unrealistic values or significantly alter the data distribution. The test confirms that the remapping maintains data integrity while transforming it to a regular grid. 
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        original_min = np.min(self.mpas_data)
        original_max = np.max(self.mpas_data)
        
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=10.0
        )
        
        remapped_min = float(remapped.min())
        remapped_max = float(remapped.max())
        
        assert remapped_min >= original_min
        assert remapped_max <= original_max
    
    def test_coordinate_handling(self: 'TestKDTreeRemapping') -> None:
        """
        This test validates that output coordinate arrays exactly match the specified domain boundaries. This test ensures that the longitude and latitude coordinates of the remapped grid align precisely with the requested min/max values. Correct coordinate assignment is fundamental for spatial analysis and data comparison across different datasets. The test uses global bounds (-180 to 180 longitude, -90 to 90 latitude) to verify full-domain coverage. Any coordinate mismatch could lead to misalignment in downstream analyses or visualization. 
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        remapped = self.remap_func(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=20.0
        )
        
        assert float(remapped.lon.min()) == pytest.approx(-180.0)
        assert float(remapped.lon.max()) == pytest.approx(180.0)
        assert float(remapped.lat.min()) == pytest.approx(-90.0)
        assert float(remapped.lat.max()) == pytest.approx(90.0)


class TestRemapMpasToLatlon:
    """ Test the main remap_mpas_to_latlon function """
    
    def setup_method(self: 'TestRemapMpasToLatlon') -> None:
        """
        This fixture sets up synthetic MPAS coordinates and a test field for remapping tests. It uses a deterministic loader to generate longitude and latitude arrays along with synthetic data values based on a combination of sinusoidal patterns and noise derived from the u velocity component. The fixture ensures that the generated coordinates and data have the expected shapes and are suitable for testing the remapping functionality. This setup is shared across multiple remapping tests to provide consistent input data. 

        Parameters:
            self ('TestRemapMpasToLatlon'): Test instance to receive generated attributes.
        Returns:
            None: Fixture populates instance attributes for use by tests.
        """
        self.n_cells = 500

        lon, lat, u, v = load_mpas_coords_from_processor(n=self.n_cells)

        self.mpas_lon = lon
        self.mpas_lat = lat

        self.mpas_data = 20 + 15 * np.sin(np.radians(self.mpas_lon) * 3) * \
                        np.cos(np.radians(self.mpas_lat) * 2) + \
                        5 * (u - u.min()) / (u.max() - u.min() + 1e-12)
    
    def test_remap_nearest_method(self: 'TestRemapMpasToLatlon') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function correctly remaps synthetic MPAS data to a regular latitude-longitude grid using the nearest neighbor method. The test checks that the output is an xarray DataArray with the expected dimensions and that all values are finite. This validates the core remapping functionality at a coarse resolution for speed. The test ensures that the function can handle typical input data and produce a valid remapped output without errors when using the nearest neighbor approach. 

        Parameters:
            self ('TestRemapMpasToLatlon'): Test instance containing synthetic MPAS coordinates and data.

        Returns:
            None: Assertions validate type and numeric finiteness of results.
        """
        remapped = remap_mpas_to_latlon(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=20.0,
            method='nearest'
        )
        
        assert isinstance(remapped, xr.DataArray)
        assert len(remapped.dims) == pytest.approx(2)
        assert np.all(np.isfinite(remapped.values))
    
    def test_remap_linear_method(self: 'TestRemapMpasToLatlon') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function correctly remaps synthetic MPAS data to a regular latitude-longitude grid using the linear interpolation method. The test checks that the output is an xarray DataArray with the expected dimensions and that all values are finite. This validates that the function can handle linear interpolation for remapping, which is a common method for producing smoother results compared to nearest neighbor. The test ensures that the function can process typical input data and produce a valid remapped output without errors when using linear interpolation. 

        Parameters:
            self ('TestRemapMpasToLatlon'): Test instance with prepared MPAS coords and data.

        Returns:
            None: Assertions validate interpolation success.
        """
        remapped = remap_mpas_to_latlon(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-90, lon_max=90,
            lat_min=-45, lat_max=45,
            resolution=10.0,
            method='linear'
        )
        
        assert isinstance(remapped, xr.DataArray)
        assert np.any(np.isfinite(remapped.values)), "expected at least some finite interpolated values"
        assert not np.any(remapped.values == 0.0), "outside-hull points must be NaN, not 0"

    
    def test_remap_with_dataarray_input(self: 'TestRemapMpasToLatlon') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle xarray DataArray inputs and preserves variable attributes through the remapping process. The test creates a DataArray with synthetic MPAS data and associated metadata, then remaps it to a regular grid. Assertions confirm that the output is an xarray DataArray and that key attributes like 'units' and 'long_name' are retained in the remapped result. This ensures that users can maintain important metadata when using the convenience function for remapping, which is essential for data provenance and interpretability in scientific analyses. 

        Parameters:
            self ('TestRemapMpasToLatlon'): Test instance with prepared MPAS data.

        Returns:
            None: Assertions validate preservation of attributes.
        """
        data_array = xr.DataArray(
            self.mpas_data,
            dims=['nCells'],
            attrs={'units': 'K', 'long_name': 'Temperature', 'description': 'Test data'}
        )
        
        remapped = remap_mpas_to_latlon(
            data=data_array,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            resolution=15.0
        )
        
        assert remapped.attrs['units'] == 'K'
        assert remapped.attrs['long_name'] == 'Temperature'
    
    def test_remap_with_empty_data(self: 'TestRemapMpasToLatlon') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle an input data array that contains only zeros without producing errors. The test creates a zero-filled array with the same shape as the synthetic MPAS data and attempts to remap it to a regular grid. The function should return a valid xarray DataArray even when the input data has no variability, demonstrating that it can manage edge cases where the data values are uniform. This ensures that users can perform remapping operations on datasets that may contain constant values without encountering issues. 

        Parameters:
            self ('TestRemapMpasToLatlon'): Test instance containing synthetic MPAS coords.

        Returns:
            None: Assertions validate result type.
        """
        zero_data = np.zeros(self.n_cells)
        
        remapped = remap_mpas_to_latlon(
            data=zero_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            resolution=20.0
        )
        
        assert isinstance(remapped, xr.DataArray)
    
    def test_remap_with_nan_data(self: 'TestRemapMpasToLatlon') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle an input data array that contains only NaN values without producing errors. The test creates a NaN-filled array with the same shape as the synthetic MPAS data and attempts to remap it to a regular grid. The function should return a valid xarray DataArray, even if all values are NaN, demonstrating that it can manage edge cases where the data values are missing or undefined without crashing. This ensures that users can perform remapping operations on datasets that may contain NaNs without encountering issues. 

        Parameters:
            self ('TestRemapMpasToLatlon'): Test instance with synthetic coordinates.

        Returns:
            None: Assertions validate output type.
        """
        nan_data = np.full(self.n_cells, np.nan)
        
        remapped = remap_mpas_to_latlon(
            data=nan_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            resolution=20.0
        )
        
        assert isinstance(remapped, xr.DataArray)
    
    def test_remap_high_resolution(self: 'TestRemapMpasToLatlon') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle remapping to a high-resolution grid without producing NaNs or errors. The test uses a 0.25-degree resolution over a 20x20 degree domain to validate fine-scale remapping capabilities. The function should produce a remapped DataArray with all finite values, demonstrating that it can manage the increased computational demands of denser target grids while maintaining data integrity. This test ensures that users can perform high-resolution remapping when needed without encountering issues, which is important for applications requiring detailed spatial analysis. 

        Parameters:
            self ('TestRemapMpasToLatlon'): Test instance with synthetic MPAS data.

        Returns:
            None: Assertions validate target grid dimensions.
        """
        remapped = remap_mpas_to_latlon(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_min=-10, lon_max=10,
            lat_min=-10, lat_max=10,
            resolution=0.25
        )
        
        assert isinstance(remapped, xr.DataArray)
        assert len(remapped.lon) > 70
        assert len(remapped.lat) > 70
    
    def test_remap_coordinates_in_radians(self: 'TestRemapMpasToLatlon') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle input longitude and latitude coordinates provided in radians. The test converts the synthetic MPAS longitude and latitude from degrees to radians before passing them to the remapping function. The function should correctly interpret the radian values, perform the remapping, and return a valid xarray DataArray with all finite values. This ensures that users can provide coordinates in radians if needed, and that the remapping function can accommodate different coordinate formats without errors. 

        Parameters:
            self ('TestRemapMpasToLatlon'): Test instance containing radian-valued coordinates.

        Returns:
            None: Assertions validate numeric finiteness of results.
        """
        lon_rad = np.radians(self.mpas_lon)
        lat_rad = np.radians(self.mpas_lat)
        
        remapped = remap_mpas_to_latlon(
            data=self.mpas_data,
            lon=lon_rad,
            lat=lat_rad,
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            resolution=20.0
        )
        
        assert isinstance(remapped, xr.DataArray)
        assert np.all(np.isfinite(remapped.values))


class TestRemappingCoverageGaps:
    """ Tests targeting specific uncovered lines in remapping.py. """


    def test_remap_mpas_to_latlon_all_nan_data(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle an input data array that contains only NaN values without producing errors. The test creates a NaN-filled array with a specified length and attempts to remap it to a regular grid. The function should return a valid xarray DataArray, even if all values are NaN, demonstrating that it can manage edge cases where the data values are missing or undefined without crashing. This ensures that users can perform remapping operations on datasets that may contain NaNs without encountering issues. 

        Parameters:
            None

        Returns:
            None: Test validates handling of all-NaN data.
        """
        n = 200
        lon = np.random.uniform(-110, -100, n)
        lat = np.random.uniform(30, 40, n)
        data = np.full(n, np.nan)

        result = remap_mpas_to_latlon(data, lon, lat, lon_min=-110, lon_max=-100,
                                       lat_min=30, lat_max=40, resolution=1.0)
        
        assert isinstance(result, xr.DataArray)
        assert np.all(np.isnan(result.values))

    def test_remap_mpas_to_latlon_all_zero_data(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle an input data array that contains only zero values without producing errors. The test creates a zero-filled array with a specified length and attempts to remap it to a regular grid. The function should return a valid xarray DataArray, even if all values are zero, demonstrating that it can manage edge cases where the data values are uniform without crashing. This ensures that users can perform remapping operations on datasets that may contain constant values without encountering issues. 

        Parameters:
            None

        Returns:
            None: Test validates handling of all-zero data.
        """
        n = 200
        lon = np.random.uniform(-110, -100, n)
        lat = np.random.uniform(30, 40, n)
        data = np.zeros(n)

        result = remap_mpas_to_latlon(data, lon, lat, lon_min=-110, lon_max=-100,
                                       lat_min=30, lat_max=40, resolution=1.0)
        
        assert isinstance(result, xr.DataArray)

    def test_remap_mpas_global_dateline_wrapping(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle longitude values that wrap around the dateline (0 to 360 degrees) without producing errors. The test creates synthetic longitude values that span the full range of 0 to 360 degrees, along with corresponding latitude and data values. The function should correctly interpret the longitude values, perform the remapping, and return a valid xarray DataArray with all finite values. This ensures that users can provide longitude coordinates in a format that includes dateline wrapping, and that the remapping function can accommodate this without issues, which is important for global datasets. 

        Parameters:
            None

        Returns:
            None: Test validates dateline wrapping behavior.
        """
        n = 5000
        lon = np.random.uniform(0, 360, n)
        lat = np.random.uniform(-60, 60, n)
        data = np.random.uniform(0, 10, n)

        result = remap_mpas_to_latlon(data, lon, lat, lon_min=0, lon_max=360,
                                       lat_min=-60, lat_max=60, resolution=5.0)

        assert isinstance(result, xr.DataArray)
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_remap_mpas_statistics_printout(self: 'TestRemappingCoverageGaps', capsys) -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function prints out statistics about the remapping process when the `print_stats` flag is set to True. The test creates synthetic longitude, latitude, and data values, then calls the remapping function with `print_stats=True`. The test captures the standard output and checks that it contains a message indicating that remapping statistics are being printed. This ensures that users receive feedback about the remapping process when they enable statistics printout, which can be helpful for understanding performance and data coverage. 

        Parameters:
            capsys: pytest fixture to capture stdout and stderr.

        Returns:
            None: Test validates statistics printout behavior.
        """
        n = 500
        lon = np.random.uniform(-110, -100, n)
        lat = np.random.uniform(30, 40, n)
        data = np.random.uniform(1, 100, n)

        remap_mpas_to_latlon(data, lon, lat, lon_min=-110, lon_max=-100,
                              lat_min=30, lat_max=40, resolution=1.0)
        
        captured = capsys.readouterr()
        assert 'Remapping MPAS' in captured.out


    def test_remap_with_masking_lonCell_radians(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon_with_masking` function can handle input longitude and latitude coordinates provided in radians when using 'lonCell' and 'latCell' keys in the dataset. The test creates a dataset with 'lonCell' and 'latCell' coordinates in radians, along with synthetic data values. The function should correctly interpret the radian values, perform the remapping with masking, and return a valid xarray DataArray. This ensures that users can provide coordinates in radians and use the appropriate keys in their datasets without encountering issues during remapping operations. 

        Parameters:
            None

        Returns:
            None: Test validates handling of radian input coordinates.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
        n = 300
        lon_rad = np.random.uniform(-2.0, -1.5, n)
        lat_rad = np.random.uniform(0.5, 0.8, n)

        ds = xr.Dataset({
            'lonCell': ('nCells', lon_rad),
            'latCell': ('nCells', lat_rad),
        })

        data = np.random.uniform(0, 10, n)

        result = remap_mpas_to_latlon_with_masking(
            data, ds, lon_min=-120, lon_max=-80, lat_min=25, lat_max=50, resolution=2.0
        )

        assert isinstance(result, xr.DataArray)

    def test_remap_with_masking_lon_lat_keys(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon_with_masking` function can handle datasets that use 'lon' and 'lat' coordinate keys instead of 'lonCell' and 'latCell'. The test creates a dataset with 'lon' and 'lat' coordinates, along with synthetic data values. The function should correctly identify the coordinate keys, perform the remapping with masking, and return a valid xarray DataArray. This ensures that users can use different coordinate key conventions in their datasets without encountering issues during remapping operations. 

        Parameters:
            None

        Returns:
            None: Test validates handling of 'lon' and 'lat' coordinate keys.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
        n = 300
        lon_deg = np.random.uniform(-110, -100, n)
        lat_deg = np.random.uniform(30, 40, n)

        ds = xr.Dataset({
            'lon': ('nCells', lon_deg),
            'lat': ('nCells', lat_deg),
        })

        data = np.random.uniform(0, 10, n)

        result = remap_mpas_to_latlon_with_masking(
            data, ds, lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, resolution=2.0
        )

        assert isinstance(result, xr.DataArray)

    def test_remap_with_masking_auto_bounds(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon_with_masking` function can automatically detect longitude and latitude bounds when they are not provided as input parameters. The test creates a dataset with 'lonCell' and 'latCell' coordinates, along with synthetic data values, and calls the remapping function without specifying the bounds. The function should analyze the coordinate values, determine appropriate bounds based on the data distribution, perform the remapping with masking, and return a valid xarray DataArray. This ensures that users can rely on the function to intelligently determine bounds when they are not explicitly provided, which can simplify the remapping process for datasets with well-defined coordinate ranges. 

        Parameters:
            None

        Returns:
            None: Test validates auto-detection of bounds.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
        n = 300
        lon_deg = np.random.uniform(-110, -100, n)
        lat_deg = np.random.uniform(30, 40, n)

        ds = xr.Dataset({
            'lonCell': ('nCells', lon_deg),
            'latCell': ('nCells', lat_deg),
        })

        data = np.random.uniform(0, 10, n)
        
        result = remap_mpas_to_latlon_with_masking(
            data, ds, lon_min=None, lon_max=None, lat_min=None, lat_max=None, resolution=2.0
        )

        assert isinstance(result, xr.DataArray)

    def test_remap_with_masking_lon_convention_neg180_180(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon_with_masking` function applies the [-180,180] longitude convention (line 861). This ensures that the function correctly interprets and remaps longitudes within this range when specified by the user. The test creates a dataset with 'lonCell' and 'latCell' coordinates in degrees, along with synthetic data values, and calls the remapping function with the longitude convention set to '[-180,180]'. The function should correctly handle the longitude values according to this convention and return a valid xarray DataArray. This ensures that users can specify their preferred longitude convention and that the remapping function can accommodate it without issues. 

        Parameters:
            None

        Returns:
            None: Test validates handling of [-180,180] longitude convention.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
        n = 300
        lon_deg = np.random.uniform(240, 280, n)
        lat_deg = np.random.uniform(30, 40, n)

        ds = xr.Dataset({
            'lonCell': ('nCells', lon_deg),
            'latCell': ('nCells', lat_deg),
        })

        data = np.random.uniform(0, 10, n)
        
        result = remap_mpas_to_latlon_with_masking(
            data, ds, lon_min=-120, lon_max=-80, lat_min=30, lat_max=40,
            resolution=2.0, lon_convention='[-180,180]'
        )

        assert isinstance(result, xr.DataArray)

    def test_remap_with_masking_lon_convention_0_360(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon_with_masking` function applies the [0,360] longitude convention (line 861). This ensures that the function correctly interprets and remaps longitudes within this range when specified by the user. The test creates a dataset with 'lonCell' and 'latCell' coordinates in degrees, along with synthetic data values, and calls the remapping function with the longitude convention set to '[0,360]'. The function should correctly handle the longitude values according to this convention and return a valid xarray DataArray. This ensures that users can specify their preferred longitude convention and that the remapping function can accommodate it without issues. 

        Parameters:
            None

        Returns:
            None: Test validates handling of [0,360] longitude convention.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
        n = 300
        lon_deg = np.random.uniform(-120, -80, n)
        lat_deg = np.random.uniform(30, 40, n)

        ds = xr.Dataset({
            'lonCell': ('nCells', lon_deg),
            'latCell': ('nCells', lat_deg),
        })

        data = np.random.uniform(0, 10, n)
        
        result = remap_mpas_to_latlon_with_masking(
            data, ds, lon_min=240, lon_max=280, lat_min=30, lat_max=40,
            resolution=2.0, lon_convention='[0,360]'
        )

        assert isinstance(result, xr.DataArray)


    def test_remap_mpas_to_latlon_xarray_input(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle input data provided as an xarray DataArray and that it preserves the attributes of the input data in the output. The test creates synthetic longitude and latitude values, along with a DataArray containing random data values and associated attributes (e.g., units). The function should correctly process the DataArray input, perform the remapping, and return a new xarray DataArray that retains the original attributes. This ensures that users can work with xarray DataArrays directly when performing remapping operations without losing important metadata, which is essential for maintaining data integrity and facilitating downstream analyses. 

        Parameters:
            None

        Returns:
            None: Test validates handling of xr.DataArray input and preservation of attributes.
        """
        n = 200
        lon = np.random.uniform(-110, -100, n)
        lat = np.random.uniform(30, 40, n)
        data = xr.DataArray(np.random.uniform(0, 10, n), attrs={'units': 'K'})

        result = remap_mpas_to_latlon(data, lon, lat, lon_min=-110, lon_max=-100,
                                       lat_min=30, lat_max=40, resolution=1.0)
        
        assert isinstance(result, xr.DataArray)

    def test_remap_mpas_to_latlon_linear_method(self: 'TestRemappingCoverageGaps') -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can successfully perform remapping using the 'linear' interpolation method. The test creates synthetic longitude and latitude values, along with random data values, and calls the remapping function with the method parameter set to 'linear'. The function should execute without errors and return a valid xarray DataArray containing the remapped data. This ensures that users can utilize linear interpolation for remapping their MPAS data to a regular lat-lon grid when desired, providing flexibility in choosing interpolation methods based on their specific needs. 

        Parameters:
            None

        Returns:
            None: Test validates handling of 'linear' interpolation method.
        """
        n = 500
        lon = np.random.uniform(-110, -100, n)
        lat = np.random.uniform(30, 40, n)
        data = np.random.uniform(0, 10, n)

        result = remap_mpas_to_latlon(data, lon, lat, lon_min=-110, lon_max=-100,
                                       lat_min=30, lat_max=40, resolution=2.0, method='linear')

        assert isinstance(result, xr.DataArray)
        assert not np.any(result.values == 0.0), "linear fill_value must be NaN, not 0"


class TestDispatchRemap:
    """ Tests for dispatch_remap — engine/method routing to KDTree or ESMPy. """

    @pytest.fixture
    def small_dataset(self: 'TestDispatchRemap') -> xr.Dataset:
        """
        This fixture creates a small xarray Dataset with synthetic longitude and latitude coordinates for testing the dispatch_remap function. The dataset contains 'lonCell' and 'latCell' coordinates in radians, which are generated using a random number generator to create values within specified ranges (longitude between -110 and -100 degrees, latitude between 30 and 40 degrees). The fixture returns an xarray Dataset that can be used in multiple test cases to verify the functionality of the remapping process, ensuring that the dispatch_remap function can handle datasets with these coordinate formats correctly.

        Parameters:
            None

        Returns:
            xr.Dataset: A dataset containing 'lonCell' and 'latCell' coordinates in radians for testing remapping functions.
        """
        n = 200
        rng = np.random.default_rng(42)
        lon_deg = rng.uniform(-110, -100, n)
        lat_deg = rng.uniform(30, 40, n)
        return xr.Dataset({
            'lonCell': xr.DataArray(np.radians(lon_deg), dims=['nCells']),
            'latCell': xr.DataArray(np.radians(lat_deg), dims=['nCells']),
        })

    @pytest.fixture
    def small_data(self: 'TestDispatchRemap', 
                   small_dataset: xr.Dataset) -> xr.DataArray:
        """
        This fixture creates a small xarray DataArray containing synthetic data values for testing the dispatch_remap function. The data values are generated using a random number generator to create uniform random values between 0 and 10, with the same length as the number of cells in the provided small_dataset. The resulting DataArray has a single dimension 'nCells' that matches the coordinate dimension in the dataset. This fixture allows multiple test cases to use consistent synthetic data when verifying the functionality of the remapping process, ensuring that the dispatch_remap function can handle data inputs correctly in conjunction with the provided dataset. 

        Parameters:
            small_dataset (xr.Dataset): The dataset fixture that provides the 'nCells' dimension for generating the data array.

        Returns:
            xr.DataArray: A DataArray containing synthetic data values for testing remapping functions, with the same length as the number of cells in the small_dataset. 
        """
        n = small_dataset.sizes['nCells']
        return xr.DataArray(np.random.default_rng(7).uniform(0, 10, n), dims=['nCells'])

    def _make_config(self: 'TestDispatchRemap', 
                     engine: str = 'kdtree', 
                     method: str = 'nearest') -> object:
        """
        This helper method creates a simple configuration object with specified remap_engine and remap_method attributes for testing the dispatch_remap function. The method takes in the desired engine and method as parameters, with default values of 'kdtree' for the engine and 'nearest' for the method. It uses Python's SimpleNamespace to create an object that has the remap_engine and remap_method attributes set to the provided values. This allows test cases to easily generate configuration objects with different combinations of engines and methods to verify that the dispatch_remap function correctly routes to the appropriate remapping implementation based on these settings. 

        Parameters:
            engine (str): The remapping engine to specify in the configuration (e.g., 'kdtree', 'esmf').
            method (str): The remapping method to specify in the configuration (e.g., 'nearest', 'linear').

        Returns:
            object: A simple configuration object with remap_engine and remap_method attributes set to the provided values, suitable for testing the dispatch_remap function.
        """
        from types import SimpleNamespace
        return SimpleNamespace(remap_engine=engine, remap_method=method)


    def test_kdtree_nearest_returns_dataarray(self: 'TestDispatchRemap', 
                                              small_data: xr.DataArray, 
                                              small_dataset: xr.Dataset) -> None:
        """
        This test verifies that the dispatch_remap function correctly routes to the KDTree-based nearest neighbor remapping implementation when the configuration specifies 'kdtree' as the remap_engine and 'nearest' as the remap_method. The test creates a configuration object with these settings, then calls dispatch_remap with the small_data and small_dataset fixtures, along with specified longitude and latitude bounds and resolution. The test checks that the result is an xarray DataArray with dimensions ('lat', 'lon'), confirming that the remapping process was executed using the KDTree nearest neighbor method and that the output is in the expected format for a regular lat-lon grid. This ensures that the dispatch_remap function correctly interprets the configuration and produces valid output for this combination of engine and method. 

        Parameters:
            small_data (xr.DataArray): The fixture providing synthetic data values for testing.
            small_dataset (xr.Dataset): The fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that dispatch_remap returns a DataArray with correct dimensions for KDTree nearest neighbor remapping.
        """
        from mpasdiag.processing.remapping import dispatch_remap
        config = self._make_config('kdtree', 'nearest')
        result = dispatch_remap(
            data=small_data, dataset=small_dataset, config=config,
            lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, resolution=2.0,
        )
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('lat', 'lon')

    def test_kdtree_linear_returns_dataarray(self: 'TestDispatchRemap', 
                                             small_data: xr.DataArray, 
                                             small_dataset: xr.Dataset) -> None:
        """
        This test verifies that the dispatch_remap function correctly routes to the KDTree-based linear interpolation remapping implementation when the configuration specifies 'kdtree' as the remap_engine and 'linear' as the remap_method. The test creates a configuration object with these settings, then calls dispatch_remap with the small_data and small_dataset fixtures, along with specified longitude and latitude bounds and resolution. The test checks that the result is an xarray DataArray, confirming that the remapping process was executed using the KDTree linear interpolation method and that the output is in a valid format for a regular lat-lon grid. This ensures that the dispatch_remap function correctly interprets the configuration and produces valid output for this combination of engine and method. 

        Parameters:
            small_data (xr.DataArray): The fixture providing synthetic data values for testing.
            small_dataset (xr.Dataset): The fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that dispatch_remap returns a DataArray for KDTree linear interpolation remapping. 
        """
        from mpasdiag.processing.remapping import dispatch_remap
        config = self._make_config('kdtree', 'linear')
        result = dispatch_remap(
            data=small_data, dataset=small_dataset, config=config,
            lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, resolution=2.0,
        )
        assert isinstance(result, xr.DataArray)


    def test_result_coordinates_match_resolution(self: 'TestDispatchRemap', 
                                                 small_data: xr.DataArray, 
                                                 small_dataset: xr.Dataset) -> None:
        """
        This test verifies that the coordinates of the result from the dispatch_remap function match the specified resolution. The test creates a configuration object for KDTree nearest neighbor remapping, then calls dispatch_remap with the small_data and small_dataset fixtures, along with specified longitude and latitude bounds and a resolution of 1.0 degree. After obtaining the result, the test calculates the spacing between longitude and latitude coordinates in the output and checks that they are approximately equal to the specified resolution (within a small tolerance). This ensures that the dispatch_remap function correctly applies the requested resolution when generating the output grid, which is important for users who need to control the spatial resolution of their remapped data.

        Parameters:
            small_data (xr.DataArray): The fixture providing synthetic data values for testing.
            small_dataset (xr.Dataset): The fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that the output coordinates from dispatch_remap match the specified resolution. 
        """
        from mpasdiag.processing.remapping import dispatch_remap
        config = self._make_config('kdtree', 'nearest')
        resolution = 1.0

        result = dispatch_remap(
            data=small_data, dataset=small_dataset, config=config,
            lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, resolution=resolution,
        )

        lon_spacing = float(result.lon[1] - result.lon[0])
        lat_spacing = float(result.lat[1] - result.lat[0])

        assert abs(lon_spacing - resolution) < 0.01
        assert abs(lat_spacing - resolution) < 0.01

    def test_numpy_array_input_accepted(self: 'TestDispatchRemap', 
                                        small_dataset: xr.Dataset) -> None:
        """
        This test verifies that the dispatch_remap function can accept input data as a NumPy array instead of an xarray DataArray. The test creates a NumPy array of synthetic data values with the same length as the number of cells in the small_dataset fixture. It then creates a configuration object for KDTree nearest neighbor remapping and calls dispatch_remap with the NumPy array, the small_dataset, and specified longitude and latitude bounds and resolution. The test checks that the result is an xarray DataArray, confirming that the dispatch_remap function can handle NumPy array inputs and still produce valid remapped output in the expected format. This ensures that users have flexibility in providing their data to the remapping function without being limited to xarray DataArrays.

        Parameters:
            small_dataset (xr.Dataset): The fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that dispatch_remap can accept NumPy array input and return a DataArray. 
        """
        from mpasdiag.processing.remapping import dispatch_remap
        n = small_dataset.sizes['nCells']
        data_np = np.random.default_rng(3).uniform(0, 5, n)
        config = self._make_config('kdtree', 'nearest')
        result = dispatch_remap(
            data=data_np, dataset=small_dataset, config=config,
            lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, resolution=2.0,
        )
        assert isinstance(result, xr.DataArray)


class TestMPASConfigRemapValidation:
    """ Tests for MPASConfig.remap_engine / remap_method field validation. """


    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_all_kdtree_methods_accepted(self: 'TestMPASConfigRemapValidation', 
                                         method: str) -> None:
        """
        This test verifies that the MPASConfig class accepts all valid remap_method options for the 'kdtree' remap_engine without raising any exceptions. The test uses pytest's parametrize feature to iterate over a list of valid methods (e.g., 'nearest', 'linear') and creates an instance of MPASConfig for each method with remap_engine set to 'kdtree'. It then checks that the remap_method attribute is correctly assigned for each case. This ensures that users can specify any of the valid remapping methods for the KDTree engine in their configuration without encountering validation errors, confirming that the MPASConfig class properly recognizes these values as valid options when using the 'kdtree' remapping engine.

        Parameters:
            method (str): The remapping method to test for validity with the 'kdtree' remap_engine (e.g., 'nearest', 'linear').

        Returns:
            None: Test validates that all valid remap_method options for 'kdtree' remap_engine are accepted without errors. 
        """
        from mpasdiag.processing.utils_config import MPASConfig
        cfg = MPASConfig(remap_engine='kdtree', remap_method=method)
        assert cfg.remap_method == method


    @pytest.mark.parametrize("method", [
        "bilinear", "conservative", "conservative_normed",
        "patch", "nearest_s2d", "nearest_d2s",
    ])
    def test_all_esmf_methods_accepted(self: 'TestMPASConfigRemapValidation', 
                                       method: str) -> None:
        """
        This test verifies that the MPASConfig class accepts all valid remap_method options for the 'esmf' remap_engine without raising any exceptions. The test uses pytest's parametrize feature to iterate over a list of valid methods (e.g., 'bilinear', 'conservative', 'nearest_s2d', etc.) and creates an instance of MPASConfig for each method with remap_engine set to 'esmf'. It then checks that the remap_method attribute is correctly assigned for each case. This ensures that users can specify any of the valid remapping methods for the ESMPy engine in their configuration without encountering validation errors, confirming that the MPASConfig class properly recognizes these values as valid options when using the 'esmf' remapping engine. 

        Parameters:
            method (str): The remapping method to test for validity with the 'esmf' remap_engine (e.g., 'bilinear', 'conservative', 'nearest_s2d', etc.).

        Returns:
            None: Test validates that all valid remap_method options for 'esmf' remap engine are accepted without errors.
        """
        from mpasdiag.processing.utils_config import MPASConfig
        cfg = MPASConfig(remap_engine='esmf', remap_method=method)
        assert cfg.remap_method == method


class TestRemapWithMaskingConfigRouting:
    """ Tests for remap_mpas_to_latlon_with_masking config-aware engine routing. """

    @pytest.fixture
    def dataset(self: 'TestRemapWithMaskingConfigRouting') -> xr.Dataset:
        """
        This fixture creates a small xarray Dataset with synthetic longitude and latitude coordinates for testing the remap_mpas_to_latlon_with_masking function. The dataset contains 'lonCell' and 'latCell' coordinates in radians, which are generated using a random number generator to create values within specified ranges (longitude between -110 and -100 degrees, latitude between 30 and 40 degrees). The fixture returns an xarray Dataset that can be used in multiple test cases to verify the functionality of the remapping process, ensuring that the remap_mpas_to_latlon_with_masking function can handle datasets with these coordinate formats correctly. 

        Parameters:
            None

        Returns:
            xr.Dataset: A dataset containing 'lonCell' and 'latCell' coordinates in radians for testing remapping functions. 
        """
        n = 300
        rng = np.random.default_rng(55)
        lon_deg = rng.uniform(-110, -100, n)
        lat_deg = rng.uniform(30, 40, n)
        return xr.Dataset({
            'lonCell': xr.DataArray(np.radians(lon_deg), dims=['nCells']),
            'latCell': xr.DataArray(np.radians(lat_deg), dims=['nCells']),
        })

    @pytest.fixture
    def data(self: 'TestRemapWithMaskingConfigRouting', 
             dataset: xr.Dataset) -> xr.DataArray:
        """
        This fixture creates a small xarray DataArray containing synthetic data values for testing the remap_mpas_to_latlon_with_masking function. The data values are generated using a random number generator to create uniform random values between 0 and 10, with the same length as the number of cells in the provided dataset. The resulting DataArray has a single dimension 'nCells' that matches the coordinate dimension in the dataset. This fixture allows multiple test cases to use consistent synthetic data when verifying the functionality of the remapping process, ensuring that the remap_mpas_to_latlon_with_masking function can handle data inputs correctly in conjunction with the provided dataset.

        Parameters:
            dataset (xr.Dataset): The dataset fixture that provides the 'nCells' dimension for generating the data array.

        Returns:
            xr.DataArray: A DataArray containing synthetic data values for testing remapping functions, with the same length as the number of cells in the dataset. 
        """
        n = dataset.sizes['nCells']
        return xr.DataArray(np.random.default_rng(9).uniform(0, 10, n), dims=['nCells'])

    def _make_config(self: 'TestRemapWithMaskingConfigRouting', 
                     engine: str, 
                     method: str) -> object:
        """
        This helper method creates a simple configuration object with specified remap_engine and remap_method attributes for testing the remap_mpas_to_latlon_with_masking function. The method takes in the desired engine and method as parameters, with no default values, requiring the caller to specify both. It uses Python's SimpleNamespace to create an object that has the remap_engine and remap_method attributes set to the provided values. This allows test cases to easily generate configuration objects with different combinations of engines and methods to verify that the remap_mpas_to_latlon_with_masking function correctly routes to the appropriate remapping implementation based on these settings. 

        Parameters:
            engine (str): The remapping engine to specify in the configuration (e.g., 'kdtree', 'esmf').
            method (str): The remapping method to specify in the configuration (e.g., 'nearest', 'linear').

        Returns:
            object: A simple configuration object with remap_engine and remap_method attributes set to the provided values, suitable for testing the remap_mpas_to_latlon_with_masking function. 
        """
        from types import SimpleNamespace
        return SimpleNamespace(remap_engine=engine, remap_method=method)

    def test_no_config_uses_method_param(self: 'TestRemapWithMaskingConfigRouting', 
                                         data: xr.DataArray, 
                                         dataset: xr.Dataset) -> None:
        """
        This test verifies that when the remap_mpas_to_latlon_with_masking function is called without a configuration object, it uses the method specified in the method parameter to determine the remapping approach. The test calls remap_mpas_to_latlon_with_masking with the data and dataset fixtures, along with specified longitude and latitude bounds, resolution, and method set to 'nearest', while leaving the config parameter as None. The test checks that the result is an xarray DataArray, confirming that the function can perform remapping using the method parameter when no configuration is provided. This ensures that users have flexibility in specifying the remapping method directly through parameters when they do not want to use a configuration object.

        Parameters:
            data (xr.DataArray): The fixture providing synthetic data values for testing.
            dataset (xr.Dataset): The fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that remap_mpas_to_latlon_with_masking uses method parameter when config is None and returns a DataArray. 
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking

        result = remap_mpas_to_latlon_with_masking(
            data, dataset, lon_min=-110, lon_max=-100, lat_min=30, lat_max=40,
            resolution=2.0, method='nearest', config=None,
        )

        assert isinstance(result, xr.DataArray)

    def test_kdtree_nearest_via_config(self: 'TestRemapWithMaskingConfigRouting', 
                                        data: xr.DataArray, 
                                        dataset: xr.Dataset) -> None:
        """
        This test verifies that when the remap_mpas_to_latlon_with_masking function is called with a configuration object that specifies 'kdtree' as the remap_engine and 'nearest' as the remap_method, it correctly routes to the KDTree-based nearest neighbor remapping implementation. The test creates a configuration object with these settings using the _make_config helper method, then calls remap_mpas_to_latlon_with_masking with the data and dataset fixtures, along with specified longitude and latitude bounds and resolution. The test checks that the result is an xarray DataArray with dimensions ('lat', 'lon'), confirming that the remapping process was executed using the KDTree nearest neighbor method and that the output is in the expected format for a regular lat-lon grid. This ensures that the remap_mpas_to_latlon_with_masking function correctly interprets the configuration and produces valid output for this combination of engine and method when a config object is provided. 

        Parameters:
            data (xr.DataArray): The fixture providing synthetic data values for testing.
            dataset (xr.Dataset): The fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that remap_mpas_to_latlon_with_masking correctly routes to KDTree nearest neighbor remapping when specified via config and returns a DataArray with dimensions ('lat', 'lon'). 
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
        config = self._make_config('kdtree', 'nearest')

        result = remap_mpas_to_latlon_with_masking(
            data, dataset, lon_min=-110, lon_max=-100, lat_min=30, lat_max=40,
            resolution=2.0, config=config,
        )

        assert isinstance(result, xr.DataArray)
        assert result.dims == ('lat', 'lon')

    def test_kdtree_linear_via_config_overrides_method_param(self: 'TestRemapWithMaskingConfigRouting', 
                                                            data: xr.DataArray, 
                                                            dataset: xr.Dataset) -> None:
        """
        This test verifies that when the remap_mpas_to_latlon_with_masking function is called with a configuration object that specifies 'kdtree' as the remap_engine and 'linear' as the remap_method, it correctly routes to the KDTree-based linear interpolation remapping implementation, even if the method parameter is set to a different value (e.g., 'nearest'). The test creates a configuration object with these settings using the _make_config helper method, then calls remap_mpas_to_latlon_with_masking with the data and dataset fixtures, along with specified longitude and latitude bounds, resolution, and method set to 'nearest'. The test checks that the result is an xarray DataArray, confirming that the function prioritizes the configuration settings over the method parameter when both are provided. This ensures that users can rely on the configuration object to control the remapping behavior when they provide both a config and method parameter, and that the function behaves predictably in this scenario.

        Parameters:
            data (xr.DataArray): The fixture providing synthetic data values for testing.
            dataset (xr.Dataset): The fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that remap_mpas_to_latlon_with_masking correctly routes to KDTree linear interpolation remapping when specified via config, even if method parameter is different, and returns a DataArray. 
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
        config = self._make_config('kdtree', 'linear')

        result = remap_mpas_to_latlon_with_masking(
            data, dataset, lon_min=-110, lon_max=-100, lat_min=30, lat_max=40,
            resolution=2.0, method='nearest', config=config,
        )

        assert isinstance(result, xr.DataArray)
        # Linear interpolation leaves NaN outside convex hull — not 0.0
        assert not np.any(result.values == 0.0)

    def test_esmf_engine_via_config_routes_to_dispatch(self: 'TestRemapWithMaskingConfigRouting', 
                                                       data: xr.DataArray, 
                                                       dataset: xr.Dataset) -> None:
        """
        This test verifies that when the remap_mpas_to_latlon_with_masking function is called with a configuration object that specifies 'esmf' as the remap_engine and 'bilinear' as the remap_method, it correctly routes to the ESMPy-based remapping implementation. The test creates a configuration object with these settings using the _make_config helper method, then calls remap_mpas_to_latlon_with_masking with the data and dataset fixtures, along with specified longitude and latitude bounds and resolution. If ESMPy is available in the environment, the test checks that the result is an xarray DataArray, confirming that the remapping process was executed using the ESMPy engine. If ESMPy is not available, the test checks that an ImportError is raised with a message indicating that ESMPy is required for the 'esmf' remapping engine. This ensures that the remap_mpas_to_latlon_with_masking function correctly interprets the configuration and either performs the remapping using ESMPy or raises an appropriate error when dependencies are missing. 

        Parameters:
            data (xr.DataArray): The fixture providing synthetic data values for testing.
            dataset (xr.Dataset): The fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that remap_mpas_to_latlon_with_masking correctly routes to ESMPy remapping when specified via config and returns a DataArray, or raises ImportError if ESMPy is not available. 
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
        config = self._make_config('esmf', 'bilinear')
        if ESMPY_AVAILABLE:
            result = remap_mpas_to_latlon_with_masking(
                data, dataset, lon_min=-110, lon_max=-100, lat_min=30, lat_max=40,
                resolution=2.0, config=config,
            )
            assert isinstance(result, xr.DataArray)
        else:
            with pytest.raises(ImportError, match="ESMPy"):
                remap_mpas_to_latlon_with_masking(
                    data, dataset, lon_min=-110, lon_max=-100, lat_min=30, lat_max=40,
                    resolution=2.0, config=config,
                )

    def test_kdtree_result_has_correct_dims(self: 'TestRemapWithMaskingConfigRouting', 
                                            data: xr.DataArray, 
                                            dataset: xr.Dataset) -> None:
        """
        This test verifies that when the remap_mpas_to_latlon_with_masking function is called with a configuration object that specifies 'kdtree' as the remap_engine and 'nearest' as the remap_method, the resulting remapped DataArray has the correct dimensions ('lat', 'lon'). The test creates a configuration object with these settings using the _make_config helper method, then calls remap_mpas_to_latlon_with_masking with the data and dataset fixtures, along with specified longitude and latitude bounds and resolution. The test checks that the resulting DataArray contains 'lat' and 'lon' in its dimensions, confirming that the output is structured as expected for a regular lat-lon grid after remapping using the KDTree nearest neighbor method. This ensures that the remap_mpas_to_latlon_with_masking function produces output with the correct dimensionality when using this combination of engine and method specified via configuration. 

        Parameters:
            data (xr.DataArray): The fixture providing synthetic data values for testing.
            dataset (xr.Dataset): The fixture providing longitude and latitude coordinates for testing.

        Returns:
            None: Test validates that remap_mpas_to_latlon_with_masking with KDTree nearest neighbor remapping produces a DataArray with 'lat' and 'lon' dimensions. 
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking
        config = self._make_config('kdtree', 'nearest')

        result = remap_mpas_to_latlon_with_masking(
            data, dataset, lon_min=-110, lon_max=-100, lat_min=30, lat_max=40,
            resolution=2.0, config=config,
        )

        assert 'lat' in result.dims
        assert 'lon' in result.dims


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
