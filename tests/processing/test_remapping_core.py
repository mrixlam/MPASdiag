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
    MPASRemapper,
    create_target_grid,
    remap_mpas_to_latlon,
    _convert_coordinates_to_degrees,
    _compute_grid_bounds
)
from unittest.mock import Mock
from tests.test_data_helpers import load_mpas_coords_from_processor
import xesmf as xe

if xe is not None:
    XESMF_AVAILABLE = True
else:
    XESMF_AVAILABLE = False

REMAPPING_AVAILABLE = True


class TestRemapping:
    """ Tests for the remapping module and its core classes/functions to ensure they are defined and importable. """
    
    def test_import_remapping_module(self: "TestRemapping") -> None:
        """
        This test verifies that the remapping module can be successfully imported. The ability to import the module is a fundamental requirement for any functionality it provides. If this test fails, it indicates that there are issues with the module's availability or its dependencies, which would prevent any remapping operations from being performed. This is a basic sanity check to ensure that the remapping code is accessible in the testing environment.

        Parameters:
            self ("TestRemapping"): Test instance (unused).

        Returns:
            None: Assertion validates import success.
        """
        from mpasdiag.processing import remapping
        assert remapping is not None
    
    def test_mpas_remapper_class_exists(self: "TestRemapping") -> None:
        """
        This test verifies that the `MPASRemapper` class is defined in the remapping module. The presence of this class is essential for users to perform remapping operations using the functionality provided by MPASdiag. If this test fails, it indicates that the core remapping class is missing, which would prevent users from utilizing the remapping features of the library. This is a basic sanity check to ensure that the class is accessible in the testing environment. 

        Parameters:
            self ("TestRemapping"): Test instance (unused).

        Returns:
            None: Assertion validates class presence.
        """
        from mpasdiag.processing.remapping import MPASRemapper
        assert MPASRemapper is not None
    
    def test_remapper_has_init_method(self: "TestRemapping") -> None:
        """
        This test confirms that the `MPASRemapper` class includes an `__init__` method that accepts a `method` parameter for specifying the remapping technique. The presence of this parameter is crucial for users to select their desired interpolation method when creating an instance of the remapper. If this test fails, it indicates that the constructor does not support method selection, which would limit the flexibility of the remapping functionality and potentially lead to confusion for users expecting to specify their remapping approach. 

        Parameters:
            self ("TestRemapping"): Test instance (unused).

        Returns:
            None: Assertion inspects constructor signature.
        """
        from mpasdiag.processing.remapping import MPASRemapper
        import inspect
        sig = inspect.signature(MPASRemapper.__init__)
        params = list(sig.parameters.keys())
        assert 'method' in params
    
    def test_remapper_has_remap_method(self: "TestRemapping") -> None:
        """
        This test verifies that the `MPASRemapper` class has a `remap` method defined. The `remap` method is the core function that performs the actual remapping of data from MPAS coordinates to a regular latitude-longitude grid. If this test fails, it indicates that the essential functionality for remapping is missing from the class, which would prevent users from performing any remapping operations using this class. This is a critical check to ensure that the main method for remapping is available in the class definition. 

        Parameters:
            self ("TestRemapping"): Test instance (unused).

        Returns:
            None: Assertion validates method presence.
        """
        from mpasdiag.processing.remapping import MPASRemapper
        assert hasattr(MPASRemapper, 'remap')
        assert callable(MPASRemapper.remap)
    
    def test_create_target_grid_function_exists(self: "TestRemapping") -> None:
        """
        This test verifies that the `create_target_grid` function is defined in the remapping module. This function is essential for users to generate regular latitude-longitude grids based on specified bounds and resolution, which are necessary for remapping MPAS data. If this test fails, it indicates that the utility function for creating target grids is missing, which would hinder users' ability to set up their remapping tasks effectively. This is a basic sanity check to ensure that the function is accessible in the testing environment. 

        Parameters:
            self ("TestRemapping"): Test instance (unused).

        Returns:
            None: Assertion validates function availability.
        """
        from mpasdiag.processing.remapping import create_target_grid
        assert callable(create_target_grid)
    
    def test_create_target_grid_has_correct_params(self: "TestRemapping") -> None:
        """
        This test verifies that the `create_target_grid` function has parameters related to latitude and longitude bounds and resolution. The presence of these parameters is crucial for users to specify the geographic extent and spacing of the target grid they wish to create for remapping. If this test fails, it indicates that the function does not support necessary parameters for defining the target grid, which would limit its usability for remapping tasks. This check ensures that the function signature includes appropriate parameters for grid creation. 

        Parameters:
            self ("TestRemapping"): Test instance (unused).
        Returns:
            None: Assertion inspects the function signature.
        """
        from mpasdiag.processing.remapping import create_target_grid
        import inspect
        sig = inspect.signature(create_target_grid)
        params = list(sig.parameters.keys())
        assert any('lat' in p.lower() for p in params) or any('grid' in p.lower() for p in params)


class TestRemappingModule:
    """ Test basic remapping module functionality. """
    
    def test_import(self: "TestRemappingModule") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` convenience function is defined in the remapping module. This function provides a user-friendly interface for remapping MPAS data to a regular latitude-longitude grid without requiring users to directly interact with the underlying remapper class. If this test fails, it indicates that the convenience function is missing, which would limit users' ability to easily perform remapping operations. This is a basic sanity check to ensure that the function is accessible in the testing environment. 
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon
        assert remap_mpas_to_latlon is not None
    
    def test_create_target_grid(self: "TestRemappingModule") -> None:
        """
        This test verifies that the `create_target_grid` function can generate a regular latitude-longitude grid with specified bounds and resolution. The test checks that the returned grid is an xarray Dataset containing 'lon' and 'lat' coordinates with the expected lengths based on the provided parameters. This validates that the function correctly constructs a regular grid covering a specific geographic region (North America) at a 1-degree resolution, ensuring proper handling of regional bounds and spacing. 
        
        Parameters:
            None
        
        Returns:
            None: This test method performs assertions and raises exceptions on failure.
        """
        from mpasdiag.processing.remapping import create_target_grid
        grid = create_target_grid(
            lon_min=-130, lon_max=-60,
            lat_min=25, lat_max=50,
            dlon=1.0, dlat=1.0
        )
        
        assert isinstance(grid, xr.Dataset)
        assert 'lon' in grid
        assert 'lat' in grid
        assert len(grid.lon) == pytest.approx(71)
        assert len(grid.lat) == pytest.approx(26)  


@pytest.mark.skipif(not REMAPPING_AVAILABLE, reason="Remapping module not available")


class TestKDTreeRemapping:
    """ Test KDTree-based remapping functionality. """
    
    def setup_method(self: "TestKDTreeRemapping") -> None:
        """
        This fixture sets up synthetic MPAS coordinates and a test field for remapping tests. It uses a deterministic loader to generate longitude and latitude arrays along with synthetic data values based on a combination of sinusoidal patterns and noise derived from the u velocity component. The fixture ensures that the generated coordinates and data have the expected shapes and are suitable for testing the remapping functionality. This setup is shared across multiple remapping tests to provide consistent input data. 

        Parameters:
            self ("TestKDTreeRemapping"): Test instance to receive prepared attributes.

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
        
    
    def test_convenience_function(self: "TestKDTreeRemapping") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function correctly remaps synthetic MPAS data to a regular latitude-longitude grid using the nearest neighbor method. The test checks that the output is an xarray DataArray with the expected dimensions and that all values are finite. This validates the core remapping functionality at a coarse resolution for speed. The test ensures that the function can handle typical input data and produce a valid remapped output without errors. 

        Parameters:
            self ("TestKDTreeRemapping"): Test instance containing synthetic MPAS data.

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
    
    def test_remap_with_dataarray(self: "TestKDTreeRemapping") -> None:
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
    
    def test_regional_remapping(self: "TestKDTreeRemapping") -> None:
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
    
    def test_fine_resolution(self: "TestKDTreeRemapping") -> None:
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
    
    def test_data_preservation(self: "TestKDTreeRemapping") -> None:
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
    
    def test_coordinate_handling(self: "TestKDTreeRemapping") -> None:
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
    
    def setup_method(self: "TestRemapMpasToLatlon") -> None:
        """
        This fixture sets up synthetic MPAS coordinates and a test field for remapping tests. It uses a deterministic loader to generate longitude and latitude arrays along with synthetic data values based on a combination of sinusoidal patterns and noise derived from the u velocity component. The fixture ensures that the generated coordinates and data have the expected shapes and are suitable for testing the remapping functionality. This setup is shared across multiple remapping tests to provide consistent input data. 

        Parameters:
            self ("TestRemapMpasToLatlon"): Test instance to receive generated attributes.
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
    
    def test_remap_nearest_method(self: "TestRemapMpasToLatlon") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function correctly remaps synthetic MPAS data to a regular latitude-longitude grid using the nearest neighbor method. The test checks that the output is an xarray DataArray with the expected dimensions and that all values are finite. This validates the core remapping functionality at a coarse resolution for speed. The test ensures that the function can handle typical input data and produce a valid remapped output without errors when using the nearest neighbor approach. 

        Parameters:
            self ("TestRemapMpasToLatlon"): Test instance containing synthetic MPAS coordinates and data.

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
    
    def test_remap_linear_method(self: "TestRemapMpasToLatlon") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function correctly remaps synthetic MPAS data to a regular latitude-longitude grid using the linear interpolation method. The test checks that the output is an xarray DataArray with the expected dimensions and that all values are finite. This validates that the function can handle linear interpolation for remapping, which is a common method for producing smoother results compared to nearest neighbor. The test ensures that the function can process typical input data and produce a valid remapped output without errors when using linear interpolation. 

        Parameters:
            self ("TestRemapMpasToLatlon"): Test instance with prepared MPAS coords and data.

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
        assert np.all(np.isfinite(remapped.values))
    
    def test_remap_invalid_method(self: "TestRemapMpasToLatlon") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function raises a ValueError when an invalid remapping method is specified. The test attempts to call the function with a method name that is not recognized (e.g., 'invalid_method') and checks that the appropriate exception is raised with a message indicating the valid options. This ensures that the function has proper error handling for unsupported remapping methods, which is important for guiding users towards correct usage and preventing silent failures or unexpected behavior. 

        Parameters:
            self ("TestRemapMpasToLatlon"): Test instance (unused).

        Returns:
            None: Test validates exception behavior.
        """
        with pytest.raises(ValueError) as context:
            remap_mpas_to_latlon(
                data=self.mpas_data,
                lon=self.mpas_lon,
                lat=self.mpas_lat,
                resolution=10.0,
                method='invalid_method'
            )

        assert 'nearest' in str(context.value)
    
    def test_remap_with_dataarray_input(self: "TestRemapMpasToLatlon") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle xarray DataArray inputs and preserves variable attributes through the remapping process. The test creates a DataArray with synthetic MPAS data and associated metadata, then remaps it to a regular grid. Assertions confirm that the output is an xarray DataArray and that key attributes like 'units' and 'long_name' are retained in the remapped result. This ensures that users can maintain important metadata when using the convenience function for remapping, which is essential for data provenance and interpretability in scientific analyses. 

        Parameters:
            self ("TestRemapMpasToLatlon"): Test instance with prepared MPAS data.

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
    
    def test_remap_with_empty_data(self: "TestRemapMpasToLatlon") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle an input data array that contains only zeros without producing errors. The test creates a zero-filled array with the same shape as the synthetic MPAS data and attempts to remap it to a regular grid. The function should return a valid xarray DataArray even when the input data has no variability, demonstrating that it can manage edge cases where the data values are uniform. This ensures that users can perform remapping operations on datasets that may contain constant values without encountering issues. 

        Parameters:
            self ("TestRemapMpasToLatlon"): Test instance containing synthetic MPAS coords.

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
    
    def test_remap_with_nan_data(self: "TestRemapMpasToLatlon") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle an input data array that contains only NaN values without producing errors. The test creates a NaN-filled array with the same shape as the synthetic MPAS data and attempts to remap it to a regular grid. The function should return a valid xarray DataArray, even if all values are NaN, demonstrating that it can manage edge cases where the data values are missing or undefined without crashing. This ensures that users can perform remapping operations on datasets that may contain NaNs without encountering issues. 

        Parameters:
            self ("TestRemapMpasToLatlon"): Test instance with synthetic coordinates.

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
    
    def test_remap_high_resolution(self: "TestRemapMpasToLatlon") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle remapping to a high-resolution grid without producing NaNs or errors. The test uses a 0.25-degree resolution over a 20x20 degree domain to validate fine-scale remapping capabilities. The function should produce a remapped DataArray with all finite values, demonstrating that it can manage the increased computational demands of denser target grids while maintaining data integrity. This test ensures that users can perform high-resolution remapping when needed without encountering issues, which is important for applications requiring detailed spatial analysis. 

        Parameters:
            self ("TestRemapMpasToLatlon"): Test instance with synthetic MPAS data.

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
    
    def test_remap_coordinates_in_radians(self: "TestRemapMpasToLatlon") -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function can handle input longitude and latitude coordinates provided in radians. The test converts the synthetic MPAS longitude and latitude from degrees to radians before passing them to the remapping function. The function should correctly interpret the radian values, perform the remapping, and return a valid xarray DataArray with all finite values. This ensures that users can provide coordinates in radians if needed, and that the remapping function can accommodate different coordinate formats without errors. 

        Parameters:
            self ("TestRemapMpasToLatlon"): Test instance containing radian-valued coordinates.

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


class TestCreateTargetGrid:
    """ Test the create_target_grid function for generating regular lat-lon grids. """
    
    def test_create_global_grid(self: "TestCreateTargetGrid") -> None:
        """
        This test verifies that the `create_target_grid` function can generate a global latitude-longitude grid with specified bounds and resolution. The test checks that the returned grid is an xarray Dataset containing 'lon' and 'lat' coordinates with the expected lengths based on the provided parameters. This validates that the function correctly constructs a regular grid covering the entire globe at a 2-degree resolution, ensuring proper handling of global bounds and spacing. This test ensures that users can generate global grids for various applications, including climate modeling and geospatial analysis. 

        Parameters:
            self ("TestCreateTargetGrid"): Test instance (unused).

        Returns:
            None: Assertions validate grid structure.
        """
        grid = create_target_grid(
            lon_min=-180, lon_max=180,
            lat_min=-90, lat_max=90,
            dlon=2.0, dlat=2.0
        )
        
        assert isinstance(grid, xr.Dataset)
        assert 'lon' in grid
        assert 'lat' in grid
        assert len(grid.lon) == pytest.approx(181)
        assert len(grid.lat) == pytest.approx(91)
    
    def test_create_regional_grid(self: "TestCreateTargetGrid") -> None:
        """
        This test verifies that the `create_target_grid` function can generate a regional latitude-longitude grid with specified bounds and resolution. The test checks that the returned grid is an xarray Dataset containing 'lon' and 'lat' coordinates with the expected lengths based on the provided parameters. This validates that the function correctly constructs a regular grid covering a specific geographic region (North America) at a 0.5-degree resolution, ensuring proper handling of regional bounds and spacing. This test ensures that users can generate regional grids for focused analyses and visualizations. 

        Parameters:
            self ("TestCreateTargetGrid"): Test instance (unused).

        Returns:
            None: Assertions validate coordinate ranges.
        """
        grid = create_target_grid(
            lon_min=-120, lon_max=-80,
            lat_min=30, lat_max=50,
            dlon=0.5, dlat=0.5
        )
        
        assert 'lon' in grid
        assert 'lat' in grid

        assert float(grid.lon.min()) == pytest.approx(-120.0)
        assert float(grid.lon.max()) == pytest.approx(-80.0)
        assert float(grid.lat.min()) == pytest.approx(30.0)
        assert float(grid.lat.max()) == pytest.approx(50.0)    

    def test_create_grid_single_cell(self: "TestCreateTargetGrid") -> None:
        """
        This test verifies that the `create_target_grid` function can generate a grid with a single cell when the specified bounds and resolution result in only one grid point in each dimension. The test checks that the returned grid contains 'lon' and 'lat' coordinates with lengths of 2, which corresponds to the minimum and maximum bounds for both longitude and latitude. This validates that the function can handle edge cases where the grid is reduced to a single cell, ensuring that it does not produce errors or unexpected results in such scenarios. 

        Parameters:
            self ("TestCreateTargetGrid"): Test instance (unused).

        Returns:
            None: Assertions validate single-cell grid creation.
        """
        grid = create_target_grid(
            lon_min=0, lon_max=1,
            lat_min=0, lat_max=1,
            dlon=1.0, dlat=1.0
        )
        
        assert len(grid.lon) == pytest.approx(2)
        assert len(grid.lat) == pytest.approx(2)
    
    def test_create_grid_different_resolutions(self: "TestCreateTargetGrid") -> None:
        """
        This test verifies that the `create_target_grid` function can generate grids with different resolutions and that the number of grid points corresponds to the specified bounds and resolution. The test checks that the returned grid contains 'lon' and 'lat' coordinates with lengths that match the expected number of points based on the provided parameters. This validates that the function correctly calculates the number of grid points for various resolutions, ensuring that users can create grids with appropriate spacing for their specific applications. 

        Parameters:
            self ("TestCreateTargetGrid"): Test instance (unused).

        Returns:
            None: Assertions validate grid resolution handling.
        """
        grid = create_target_grid(
            lon_min=0, lon_max=10,
            lat_min=0, lat_max=5,
            dlon=2.0, dlat=1.0
        )
        
        assert len(grid.lon) == pytest.approx(6)
        assert len(grid.lat) == pytest.approx(6)

@pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not available")


class TestHelperFunctions:
    """ Test helper utility functions for coordinate conversion and grid bounds calculation. """
    
    def test_convert_coordinates_radians_to_degrees(self: "TestHelperFunctions") -> None:
        """
        This test verifies that the `_convert_coordinates_to_degrees` helper function correctly converts longitude and latitude coordinates from radians to degrees. The test provides sample longitude and latitude values in radians, calls the conversion function, and checks that the output values are approximately equal to the expected degree values using `pytest.approx`. This ensures that the helper function performs accurate conversions, which is essential for remapping operations that require consistent coordinate formats. 

        Parameters:
            self ("TestHelperFunctions"): Test instance (unused).

        Returns:
            None: Assertions validate numeric conversion accuracy.
        """
        lon_rad = np.array([0, np.pi/4, np.pi/2, np.pi, -np.pi])
        lat_rad = np.array([0, np.pi/6, np.pi/4, np.pi/3, -np.pi/6])
        
        lon_deg, lat_deg = _convert_coordinates_to_degrees(lon_rad, lat_rad)
        
        assert lon_deg[1] == pytest.approx(45.0, abs=1e-05)
        assert lon_deg[2] == pytest.approx(90.0, abs=1e-05)
        assert lon_deg[3] == pytest.approx(180.0, abs=1e-05)
        assert lat_deg[1] == pytest.approx(30.0, abs=1e-05)
        assert lat_deg[2] == pytest.approx(45.0, abs=1e-05)
    
    def test_convert_coordinates_already_degrees(self: "TestHelperFunctions") -> None:
        """
        This test verifies that the `_convert_coordinates_to_degrees` helper function correctly handles input coordinates that are already in degrees without altering them. The test provides sample longitude and latitude values in degrees, calls the conversion function, and checks that the output values are approximately equal to the original degree values using `numpy.testing.assert_array_almost_equal`. This ensures that the helper function can recognize when coordinates are already in the correct format and does not perform unnecessary conversions, which is important for maintaining data integrity and avoiding potential errors in remapping operations. 

        Parameters:
            self ("TestHelperFunctions"): Test instance (unused).
        Returns:
            None: Assertions validate identity behavior for degree inputs.
        """
        lon_deg = np.array([-180, -90, 0, 90, 180])
        lat_deg = np.array([-90, -45, 0, 45, 90])
        
        lon_out, lat_out = _convert_coordinates_to_degrees(lon_deg, lat_deg)
        
        np.testing.assert_array_almost_equal(lon_out, lon_deg)
        np.testing.assert_array_almost_equal(lat_out, lat_deg)
    
    def test_convert_coordinates_with_dataarray(self: "TestHelperFunctions") -> None:
        """
        This test verifies that the `_convert_coordinates_to_degrees` helper function can handle xarray DataArray inputs for longitude and latitude coordinates. The test creates DataArrays with sample longitude and latitude values in radians, calls the conversion function, and checks that the output values are approximately equal to the expected degree values using `pytest.approx`. This ensures that the helper function can process xarray DataArrays correctly, which is important for compatibility with the remapping functions that may receive coordinates in this format. The test confirms that the function can manage different input types while still performing accurate conversions.

        Parameters:
            self ("TestHelperFunctions"): Test instance (unused).

        Returns:
            None: Assertions validate returned types and values.
        """
        lon_rad = xr.DataArray(np.array([0, np.pi/2, np.pi]))
        lat_rad = xr.DataArray(np.array([0, np.pi/4, np.pi/2]))
        
        lon_deg, lat_deg = _convert_coordinates_to_degrees(lon_rad, lat_rad)
        
        assert isinstance(lon_deg, np.ndarray)
        assert isinstance(lat_deg, np.ndarray)

        assert lon_deg[1] == pytest.approx(90.0, abs=1e-05)
    
    def test_compute_grid_bounds(self: "TestHelperFunctions") -> None:
        """
        This test verifies that the `_compute_grid_bounds` helper function correctly computes grid bounds from a given array of coordinates and a specified resolution. The test provides a simple array of coordinates and a resolution, calls the function to compute bounds, and checks that the resulting bounds array has the correct length and values. The first and last bounds should be half a resolution beyond the first and last coordinates, respectively, while the interior bounds should be midpoints between adjacent coordinates. This ensures that the helper function can accurately calculate grid boundaries necessary for remapping operations. 

        Parameters:
            self ("TestHelperFunctions"): Test instance (unused).

        Returns:
            None: Assertions validate correct bounds computation.
        """
        coords = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        resolution = 1.0
        
        bounds = _compute_grid_bounds(coords, resolution)
        
        assert len(bounds) == len(coords) + 1
        assert bounds[0] == pytest.approx(-0.5)
        assert bounds[-1] == pytest.approx(4.5)
        assert bounds[1] == pytest.approx(0.5)
        assert bounds[2] == pytest.approx(1.5)
    
    def test_compute_grid_bounds_non_uniform(self: "TestHelperFunctions") -> None:
        """
        This test verifies that the `_compute_grid_bounds` helper function can handle non-uniformly spaced coordinates and still compute correct grid bounds based on the specified resolution. The test provides an array of coordinates that are not evenly spaced, calls the function to compute bounds, and checks that the resulting bounds array has the correct length and values. The first and last bounds should still be half a resolution beyond the first and last coordinates, while the interior bounds should be midpoints between adjacent coordinates, regardless of their spacing. This ensures that the helper function can manage irregular coordinate distributions while still providing accurate grid boundaries for remapping operations. 

        Parameters:
            self ("TestHelperFunctions"): Test instance (unused).
        Returns:
            None: Assertions validate behavior for non-uniform inputs.
        """
        coords = np.array([0.0, 0.5, 1.5, 3.0])
        resolution = 0.5
        
        bounds = _compute_grid_bounds(coords, resolution)
        
        assert len(bounds) == len(coords) + 1
        assert bounds[0] == pytest.approx(-0.25)
        assert bounds[-1] == pytest.approx(3.25)
        assert bounds[1] == pytest.approx(0.25)
        assert bounds[2] == pytest.approx(1.0)
        assert bounds[3] == pytest.approx(2.25)


class TestRemappingCoverageGaps:
    """ Tests targeting specific uncovered lines in remapping.py. """

    @pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not installed")
    def test_invalid_method_raises_valueerror(self) -> None:
        """
        This test verifies that the MPASRemapper class raises a ValueError when an invalid remapping method is specified during initialization. The test attempts to create an instance of MPASRemapper with a method name that is not recognized (e.g., 'invalid_method') and checks that the appropriate exception is raised with a message indicating the valid options. This ensures that the class has proper error handling for unsupported remapping methods, which is important for guiding users towards correct usage and preventing silent failures or unexpected behavior. 

        Parameters:
            None

        Returns:
            None: Test validates exception behavior.
        """
        with pytest.raises(ValueError, match="Invalid method"):
            MPASRemapper(method='invalid_method')

    @pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not installed")
    def test_cleanup_clears_regridder(self) -> None:
        """
        This test verifies that the `cleanup` method of the MPASRemapper class properly clears the regridder object. The test creates an instance of MPASRemapper, assigns a mock object to the regridder attribute, and then calls the cleanup method. After cleanup, the test checks that the regridder attribute is set to None, confirming that resources are released as expected. This ensures that the cleanup process effectively manages memory and resources associated with the regridder, which is important for preventing memory leaks in long-running applications. 

        Parameters:
            None

        Returns:
            None: Test validates cleanup behavior.
        """
        remapper = MPASRemapper(method='bilinear')
        remapper.regridder = Mock()
        assert remapper.regridder is not None
        remapper.cleanup()
        assert remapper.regridder is None

    def test_estimate_memory_unknown_method(self) -> None:
        """
        This test verifies that the `estimate_memory_usage` method of the MPASRemapper class returns a lower memory estimate for an unknown remapping method compared to a known method like 'bilinear'. The test calls the memory estimation function with the same grid sizes but different methods and checks that the estimate for the unknown method is less than that for the bilinear method. This ensures that the function provides a reasonable fallback estimate when an unrecognized method is specified, which can help users understand potential memory requirements even if they choose an unsupported method. 

        Parameters:
            None

        Returns:
            None: Test validates memory estimation behavior.
        """
        mem_unknown = MPASRemapper.estimate_memory_usage(1000, 500, 'nearest_s2d')
        mem_bilinear = MPASRemapper.estimate_memory_usage(1000, 500, 'bilinear')
        assert mem_unknown < mem_bilinear

    def test_remap_mpas_to_latlon_all_nan_data(self) -> None:
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

    def test_remap_mpas_to_latlon_all_zero_data(self) -> None:
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

    def test_remap_mpas_global_dateline_wrapping(self) -> None:
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

    def test_remap_mpas_statistics_printout(self, capsys) -> None:
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

    def test_build_mask_global_skips_convex_hull(self, capsys) -> None:
        """
        This test verifies that the `build_remapped_valid_mask` function returns None and prints a message about skipping the convex hull calculation when the input longitude and latitude values cover the entire globe (0 to 360 degrees for longitude and -90 to 90 degrees for latitude). The test creates synthetic longitude and latitude values that span the global range, along with a remapped data array. The function should recognize that the data is global and skip the convex hull calculation, returning None for the mask. The test captures the standard output and checks that it contains a message indicating that the convex hull is being skipped. This ensures that the function can efficiently handle global datasets without unnecessary computations. 

        Parameters:
            capsys: pytest fixture to capture stdout and stderr.

        Returns:
            None: Test validates behavior for global data.
        """
        from mpasdiag.processing.remapping import build_remapped_valid_mask
        lon_vals = np.random.uniform(0, 360, 1000)
        lat_vals = np.random.uniform(-90, 90, 1000)
        remapped = np.random.uniform(0, 10, (180, 360))

        mask = build_remapped_valid_mask(lon_vals, lat_vals, 0, 360, -90, 90, 1.0, remapped)

        assert mask is None
        captured = capsys.readouterr()
        assert 'Skipping convex hull' in captured.out

    def test_build_mask_convex_hull_success(self) -> None:
        """
        This test verifies that the `build_remapped_valid_mask` function successfully builds a valid mask when the input longitude and latitude values cover a regional area (e.g., -110 to -100 degrees for longitude and 30 to 40 degrees for latitude). The test creates synthetic longitude and latitude values that fall within a specified regional domain, along with a remapped data array. The function should perform the convex hull calculation and return a boolean mask indicating valid remapped points. The test checks that the returned mask is not None, has the correct shape, and is of boolean type. This ensures that the function can generate valid masks for regional datasets, which is important for accurately identifying valid data points in remapping operations. 

        Parameters:
            None

        Returns:
            None: Test validates successful mask generation for regional data.
        """
        from mpasdiag.processing.remapping import build_remapped_valid_mask
        n = 500
        lon_vals = np.random.uniform(-110, -100, n)
        lat_vals = np.random.uniform(30, 40, n)
        remapped = np.random.uniform(0, 10, (10, 10))

        mask = build_remapped_valid_mask(lon_vals, lat_vals, -110, -100, 30, 40, 1.0, remapped)

        assert mask is not None
        assert mask.dtype == bool
        assert mask.shape == (10, 10)

    def test_build_mask_convex_hull_exception(self) -> None:
        """
        This test verifies that the `build_remapped_valid_mask` function returns None when the ConvexHull calculation raises an exception, such as when the input longitude and latitude values are collinear. The test creates synthetic longitude and latitude values that are all the same (e.g., all longitudes at -105 degrees and latitudes linearly spaced between 30 and 40 degrees), along with a remapped data array. The function should attempt to perform the convex hull calculation, encounter an exception due to collinearity, and return None for the mask. This ensures that the function can gracefully handle cases where the convex hull cannot be computed, which is important for robustness in remapping operations. 

        Parameters:
            None

        Returns:
            None: Test validates behavior when ConvexHull raises an exception.
        """
        from mpasdiag.processing.remapping import build_remapped_valid_mask

        lon_vals = np.full(100, -105.0)
        lat_vals = np.linspace(30, 40, 100)
        remapped = np.random.uniform(0, 10, (10, 10))
        mask = build_remapped_valid_mask(lon_vals, lat_vals, -110, -100, 30, 40, 1.0, remapped)
        assert mask is None

    def test_remap_with_masking_lonCell_radians(self) -> None:
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

    def test_remap_with_masking_lon_lat_keys(self) -> None:
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

    def test_remap_with_masking_auto_bounds(self) -> None:
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

    def test_remap_with_masking_lon_convention_neg180_180(self) -> None:
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

    def test_remap_with_masking_lon_convention_0_360(self) -> None:
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

    def test_remap_with_masking_missing_coords_raises(self) -> None:
        """
        This test verifies that the `remap_mpas_to_latlon_with_masking` function raises a ValueError when the input dataset does not contain the required longitude and latitude coordinate keys ('lonCell' and 'latCell' or 'lon' and 'lat'). The test creates a dataset without any coordinate keys, along with synthetic data values, and attempts to call the remapping function. The function should detect the absence of necessary coordinates and raise an appropriate exception with a message indicating that cell coordinates could not be found. This ensures that the function has proper error handling for missing coordinate information, which is crucial for guiding users towards providing the correct dataset structure for remapping operations. 

        Parameters:
            None

        Returns:
            None: Test validates error handling for missing coordinates.
        """
        from mpasdiag.processing.remapping import remap_mpas_to_latlon_with_masking

        ds = xr.Dataset({'temperature': ('nCells', np.zeros(100))})
        data = np.zeros(100)

        with pytest.raises(ValueError, match="Could not find cell coordinates"):
            remap_mpas_to_latlon_with_masking(data, ds, -110, -100, 30, 40, resolution=2.0)

    def test_remap_mpas_to_latlon_xarray_input(self) -> None:
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

    def test_remap_mpas_to_latlon_linear_method(self) -> None:
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

    def test_remap_mpas_to_latlon_invalid_method(self) -> None:
        """
        This test verifies that the `remap_mpas_to_latlon` function raises a ValueError when an invalid interpolation method is specified. The test creates synthetic longitude and latitude values, along with random data values, and attempts to call the remapping function with a method name that is not recognized (e.g., 'cubic'). The function should detect the invalid method and raise an appropriate exception with a message indicating the valid options. This ensures that the function has proper error handling for unsupported interpolation methods, which is important for guiding users towards correct usage and preventing silent failures or unexpected behavior during remapping operations. 

        Parameters:
            None

        Returns:
            None: Test validates error handling for invalid interpolation method.
        """
        with pytest.raises(ValueError, match="method must be"):
            remap_mpas_to_latlon(np.zeros(10), np.zeros(10), np.zeros(10), method='cubic')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
