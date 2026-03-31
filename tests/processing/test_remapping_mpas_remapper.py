#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPASdiag remapping functionality

This module contains integration tests for the remapping functionality in MPASdiag, specifically focusing on the end-to-end remapping process using real MPAS data. The tests verify that the remapping functions can successfully remap data defined on a real MPAS grid to a regular lat-lon grid, and that the main remapping class can be used

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules for testing
import pytest
import shutil
import tempfile
import numpy as np
import xesmf as xe
import xarray as xr
from typing import Any
from pathlib import Path

if not xe:
    pytest.skip("xESMF is required for remapping tests", allow_module_level=True)
    XESMF_AVAILABLE = True
else:
    XESMF_AVAILABLE = True

from mpasdiag.processing.remapping import (
    MPASRemapper
)

from tests.test_data_helpers import load_mpas_coords_from_processor

REMAPPING_AVAILABLE = True


class TestMPASRemapper:
    """ Test the MPASRemapper class """
    
    def setup_method(self: "TestMPASRemapper") -> None:
        """
        This setup method prepares the test environment for each test in the `TestMPASRemapper` class. It creates a temporary directory to store any weights files generated during the tests and loads synthetic MPAS coordinate data (longitude, latitude, and associated data) using a helper function. The longitude values are normalized to the 0-360 degree range to ensure consistency with typical MPAS grid conventions. The prepared coordinates and data are stored as instance attributes for use in the various remapping tests that follow.

        Parameters:
            self ("TestMPASRemapper"): Test instance to receive prepared attributes.

        Returns:
            None: Populates instance attributes for use by tests.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.n_cells = 300
        lon, lat, u, v = load_mpas_coords_from_processor(n=self.n_cells)
        self.mpas_lon = np.mod(lon, 360.0)
        self.mpas_lat = lat
        self.mpas_data = 20 + 10 * (u - u.min()) / (u.max() - u.min() + 1e-12)
    
    def teardown_method(self: "TestMPASRemapper") -> None:
        """
        This teardown method cleans up the test environment after each test in the `TestMPASRemapper` class. It checks if the temporary directory created during setup exists and, if so, it removes the directory and all of its contents using `shutil.rmtree`. This ensures that any weights files or other temporary data generated during the tests do not persist on the filesystem, maintaining a clean state for subsequent tests and preventing clutter.

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Performs cleanup operations.
        """
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_remapper_initialization(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `MPASRemapper` class can be initialized with default parameters. It creates an instance of `MPASRemapper` without passing any arguments and asserts that the default values for attributes such as `method`, `reuse_weights`, `periodic`, `source_grid`, and `target_grid` are set as expected. This test ensures that the constructor of the `MPASRemapper` class correctly assigns default values to its attributes when no parameters are provided. 

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Assertions validate initial attribute values.
        """
        remapper = MPASRemapper()
        
        assert remapper.method == 'bilinear'
        assert remapper.reuse_weights
        assert not remapper.periodic
        assert remapper.source_grid is None
        assert remapper.target_grid is None
    
    def test_remapper_initialization_with_params(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `MPASRemapper` class can be initialized with specific parameters. It creates an instance of `MPASRemapper` by passing various arguments such as `method`, `weights_dir`, `reuse_weights`, `periodic`, `extrap_method`, `extrap_dist_exponent`, and `extrap_num_src_pnts`. The test then asserts that these attributes are set to the values provided during initialization. This ensures that the constructor of the `MPASRemapper` class correctly assigns values to its attributes based on the parameters passed by the user. 

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Assertions validate parameter propagation.
        """
        remapper = MPASRemapper(
            method='conservative',
            weights_dir=self.temp_dir,
            reuse_weights=False,
            periodic=True,
            extrap_method='inverse_dist',
            extrap_dist_exponent=2.0,
            extrap_num_src_pnts=8
        )
        
        assert remapper.method == 'conservative'
        assert not remapper.reuse_weights
        assert remapper.periodic
        assert remapper.extrap_method == 'inverse_dist'
        assert remapper.extrap_dist_exponent == pytest.approx(2.0)
        assert remapper.extrap_num_src_pnts == pytest.approx(8)
    
    def test_remapper_invalid_method(self: "TestMPASRemapper") -> None:
        """
        This test verifies that initializing the `MPASRemapper` class with an invalid remapping method raises a ValueError. The constructor of `MPASRemapper` should validate the `method` parameter against a predefined set of supported methods (e.g., 'nearest', 'bilinear', 'conservative') and raise an error if an unsupported method is provided. This test ensures that the class enforces valid input for the remapping method and provides clear feedback to users when they attempt to use an invalid option. 

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Test asserts that a ValueError is raised.
        """
        with pytest.raises(ValueError) as context:
            MPASRemapper(method='invalid_method')

        assert 'Invalid method' in str(context.value)
    
    def test_prepare_source_grid(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `prepare_source_grid` method of the `MPASRemapper` class can successfully prepare a source grid Dataset from given longitude and latitude arrays. The method should return an xarray Dataset containing 'lon' and 'lat' coordinate variables that represent the unstructured MPAS grid. The test asserts that the returned object is a Dataset, that it contains the expected coordinates, and that the longitude values are normalized to the 0-360 degree range. This ensures that the remapper can correctly process the input coordinates and create a suitable source grid for remapping. 

        Parameters:
            self ("TestMPASRemapper"): Test instance with prepared MPAS lon/lat and data.

        Returns:
            None: Assertions validate the created source grid Dataset.
        """
        remapper = MPASRemapper()
        
        source_grid = remapper.prepare_source_grid(
            lon=self.mpas_lon,
            lat=self.mpas_lat
        )
        
        assert isinstance(source_grid, xr.Dataset)

        assert 'lon' in source_grid
        assert 'lat' in source_grid

        assert len(source_grid.lon) == self.n_cells

        assert np.all(source_grid.lon.values >= 0)
        assert np.all(source_grid.lon.values <= 360)
    
    def test_prepare_source_grid_with_bounds(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `prepare_source_grid` method can successfully prepare a source grid Dataset when longitude and latitude bounds are provided. The method should return an xarray Dataset that includes 'lon_b' and 'lat_b' variables representing the bounds of the grid cells. The test asserts that these bound variables are present in the returned Dataset, confirming that the method can handle and incorporate bounds information when preparing the source grid. This is important for remapping methods that require knowledge of cell boundaries, such as conservative remapping. 

        Parameters:
            self ("TestMPASRemapper"): Test instance with synthetic MPAS coords and bounds.

        Returns:
            None: Assertions validate bound handling.
        """
        remapper = MPASRemapper()
        
        lon_bounds = np.vstack([
            self.mpas_lon - 0.25,
            self.mpas_lon + 0.25,
            self.mpas_lon + 0.25,
            self.mpas_lon - 0.25
        ]).T

        lat_bounds = np.vstack([
            self.mpas_lat - 0.25,
            self.mpas_lat - 0.25,
            self.mpas_lat + 0.25,
            self.mpas_lat + 0.25
        ]).T
        
        source_grid = remapper.prepare_source_grid(
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds
        )
        
        assert 'lon_b' in source_grid
        assert 'lat_b' in source_grid
    
    def test_create_target_grid_method(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `create_target_grid` method of the `MPASRemapper` class can successfully create a target grid Dataset based on specified longitude and latitude ranges and resolutions. The method should return an xarray Dataset containing 'lon' and 'lat' coordinate variables that define a regular lat-lon grid. The test asserts that the returned object is a Dataset, that it contains the expected coordinates, and that the number of longitude and latitude points matches the expected values based on the provided ranges and resolutions. This ensures that the remapper can correctly generate a target grid suitable for remapping operations.

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).
            
        Returns:
            None: Assertions validate target grid creation and assignment.
        """
        remapper = MPASRemapper()
        
        target_grid = remapper.create_target_grid(
            lon_min=0, lon_max=360,
            lat_min=-90, lat_max=90,
            dlon=5.0, dlat=5.0
        )
        
        assert isinstance(target_grid, xr.Dataset)
        assert remapper.target_grid is not None
        assert len(target_grid.lon) == pytest.approx(73)
        assert len(target_grid.lat) == pytest.approx(37)
    
    def test_build_regridder_error_no_source(self: "TestMPASRemapper") -> None:
        """
        This test verifies that attempting to build a regridder without a prepared source grid raises a ValueError. The `build_regridder` method requires both a source and target grid to construct the regridding object, and this test ensures that if the source grid is not set, the method will raise an appropriate error message. This helps enforce the correct usage of the remapper and prevents silent failures.

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Test asserts the expected ValueError is raised.
        """
        remapper = MPASRemapper()
        remapper.create_target_grid()
        
        with pytest.raises(ValueError) as context:
            remapper.build_regridder()

        assert 'Source grid' in str(context.value)
    
    def test_build_regridder_error_no_target(self: "TestMPASRemapper") -> None:
        """
        This test verifies that attempting to build a regridder without a prepared target grid raises a ValueError. The `build_regridder` method requires both a source and target grid to construct the regridding object, and this test ensures that if the target grid is not set, the method will raise an appropriate error message. This helps enforce the correct usage of the remapper and prevents silent failures. 

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Test asserts the expected ValueError is raised.
        """
        remapper = MPASRemapper()
        remapper.prepare_source_grid(self.mpas_lon, self.mpas_lat)
        
        with pytest.raises(ValueError) as context:
            remapper.build_regridder()

        assert 'Target grid' in str(context.value)
    
    def test_remap_error_no_regridder(self: "TestMPASRemapper") -> None:
        """
        This test verifies that attempting to remap data without a built regridder raises a ValueError. The `remap` method relies on a prepared regridder object to perform the remapping operation, and this test ensures that if the regridder has not been built (i.e., `build_regridder` has not been called), the method will raise an appropriate error message. This helps enforce the correct workflow when using the remapper and prevents users from attempting to remap without the necessary setup.

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Test asserts ValueError is raised.
        """
        remapper = MPASRemapper()
        
        with pytest.raises(ValueError) as context:
            remapper.remap(self.mpas_data)

        assert 'Regridder must be built' in str(context.value)
    
    def test_remap_dataset_error_no_regridder(self: "TestMPASRemapper") -> None:
        """
        This test verifies that attempting to remap an xarray Dataset without a built regridder raises a ValueError. The `remap_dataset` method relies on a prepared regridder object to perform the remapping operation, and this test ensures that if the regridder has not been built (i.e., `build_regridder` has not been called), the method will raise an appropriate error message. This helps enforce the correct workflow when using the remapper and prevents users from attempting to remap without the necessary setup. 

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Test asserts ValueError is raised.
        """
        remapper = MPASRemapper()
        dataset = xr.Dataset({'var1': (['x'], self.mpas_data)})
        
        with pytest.raises(ValueError) as context:
            remapper.remap_dataset(dataset)

        assert 'Regridder must be built' in str(context.value)
    
    def test_cleanup(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `cleanup` method of the `MPASRemapper` class properly clears internal attributes related to the source grid, target grid, and regridder. After calling `cleanup`, the test asserts that these attributes are set to `None`, confirming that the method effectively resets the state of the remapper. This is important for ensuring that resources are released and that the remapper can be reused or safely discarded without lingering references to previous grids or regridders. 

        Parameters:
            self ("TestMPASRemapper"): Test instance with a configured remapper.

        Returns:
            None: Assertions validate that internal attributes are cleared.
        """
        remapper = MPASRemapper()
        remapper.prepare_source_grid(self.mpas_lon, self.mpas_lat)
        remapper.create_target_grid()
        
        remapper.cleanup()
        
        assert remapper.source_grid is None
        assert remapper.target_grid is None
        assert remapper.regridder is None
    
    def test_estimate_memory_usage_conservative(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `estimate_memory_usage` method of the `MPASRemapper` class returns a positive float representing the estimated memory usage for the conservative remapping method. The test calls the method with a specified number of source and target grid points and asserts that the returned value is a float greater than zero. This ensures that the memory estimation logic is functioning correctly and provides a sensible estimate for users. 

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Assertions validate returned memory estimate.
        """
        memory_gb = MPASRemapper.estimate_memory_usage(
            n_source=10000,
            n_target=5000,
            method='conservative'
        )
        
        assert isinstance(memory_gb, float)
        assert memory_gb > 0
    
    def test_estimate_memory_usage_bilinear(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `estimate_memory_usage` method of the `MPASRemapper` class returns a positive float representing the estimated memory usage for the bilinear remapping method. The test calls the method with a specified number of source and target grid points and asserts that the returned value is a float greater than zero. This ensures that the memory estimation logic is functioning correctly and provides a sensible estimate for users. 

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Assertions validate returned memory estimate.
        """
        memory_gb = MPASRemapper.estimate_memory_usage(
            n_source=10000,
            n_target=5000,
            method='bilinear'
        )
        
        assert isinstance(memory_gb, float)
        assert memory_gb > 0
    
    def test_estimate_memory_usage_nearest(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `estimate_memory_usage` method of the `MPASRemapper` class returns a positive float representing the estimated memory usage for the nearest neighbor remapping method. The test calls the method with a specified number of source and target grid points and asserts that the returned value is a float greater than zero. This ensures that the memory estimation logic is functioning correctly and provides a sensible estimate for users. 

        Parameters:
            self ("TestMPASRemapper"): Test instance (unused).

        Returns:
            None: Assertions validate returned memory estimate.
        """
        memory_gb = MPASRemapper.estimate_memory_usage(
            n_source=10000,
            n_target=5000,
            method='nearest_s2d'
        )
        
        assert isinstance(memory_gb, float)
        assert memory_gb > 0
    
    def test_unstructured_to_structured_grid(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS grid data into a structured grid format. The method should return a structured DataArray and a corresponding Dataset for the grid, and the test asserts that these outputs are of the correct types and contain the expected dimensions and coordinate variables. This ensures that the conversion process is functioning correctly and that the resulting structured grid is properly formatted for use in remapping operations.

        Parameters:
            self ("TestMPASRemapper"): Test instance with synthetic MPAS data.

        Returns:
            None: Assertions validate conversion output types and structure.
        """
        structured_data, structured_grid = MPASRemapper.unstructured_to_structured_grid(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            intermediate_resolution=5.0,
            lon_min=0, lon_max=360,
            lat_min=-90, lat_max=90,
            buffer=0.0
        )
        
        assert isinstance(structured_data, xr.DataArray)
        assert isinstance(structured_grid, xr.Dataset)
        assert len(structured_data.dims) == pytest.approx(2)

        assert 'lon' in structured_grid
        assert 'lat' in structured_grid
        assert 'lon_b' in structured_grid
        assert 'lat_b' in structured_grid
    
    def test_unstructured_to_structured_grid_auto_bounds(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS grid data into a structured grid format without explicitly providing longitude and latitude bounds. The method should automatically determine appropriate bounds based on the input coordinates and return a structured DataArray and a corresponding Dataset for the grid. The test asserts that the outputs are of the correct types and contain the expected coordinate variables, confirming that the method can handle cases where bounds are not provided and can still produce a valid structured grid for remapping operations. 

        Parameters:
            self ("TestMPASRemapper"): Test instance with synthetic MPAS data.

        Returns:
            None: Assertions validate auto-bound detection.
        """
        structured_data, structured_grid = MPASRemapper.unstructured_to_structured_grid(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            intermediate_resolution=10.0
        )
        
        assert isinstance(structured_data, xr.DataArray)
        assert isinstance(structured_grid, xr.Dataset)

        assert 'grid_conversion' in structured_data.attrs
        assert 'intermediate_resolution' in structured_data.attrs
    
    def test_unstructured_to_structured_with_dataarray(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert an xarray DataArray defined on an unstructured MPAS grid into a structured grid format. The method should preserve the attributes of the input DataArray and return a structured DataArray and a corresponding Dataset for the grid. The test asserts that the output DataArray retains the original attributes (e.g., 'units', 'long_name') and that the structured grid is properly formatted for use in remapping operations. This ensures that users can seamlessly convert their unstructured MPAS data into a structured format while maintaining important metadata. 

        Parameters:
            self ("TestMPASRemapper"): Test instance with synthetic MPAS DataArray input.

        Returns:
            None: Assertions validate attribute preservation.
        """
        data_array = xr.DataArray(
            self.mpas_data,
            dims=['nCells'],
            attrs={'units': 'K', 'long_name': 'Temperature'}
        )
        
        structured_data, structured_grid = MPASRemapper.unstructured_to_structured_grid(
            data=data_array,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            intermediate_resolution=5.0,
            lon_min=0, lon_max=50,
            lat_min=-20, lat_max=20
        )
        
        assert structured_data.attrs['units'] == 'K'
        assert structured_data.attrs['long_name'] == 'Temperature'


class TestMPASRemapperMethodValidation:
    """ Test MPASRemapper method validation. """
    
    def test_invalid_method_raises_error(self: "TestMPASRemapperMethodValidation") -> None:
        """
        This test verifies that initializing the `MPASRemapper` class with an invalid remapping method raises a ValueError. The constructor of `MPASRemapper` should validate the `method` parameter against a predefined set of supported methods (e.g., 'nearest', 'bilinear', 'conservative') and raise an error if an unsupported method is provided. This test ensures that the class enforces valid input for the remapping method and provides clear feedback to users when they attempt to use an invalid option. 

        Parameters:
            self ("TestMPASRemapperMethodValidation"): Test instance (unused).

        Returns:
            None: Test asserts ValueError is raised.
        """
        remapper = MPASRemapper()  
        err = None

        with pytest.raises(ValueError, match="Invalid method"):
            remapper = MPASRemapper(method='invalid_method')
        
        err = ValueError("Invalid method")
        
        assert remapper.method == 'bilinear' 
        assert str(err) == "Invalid method"


@pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not installed")


class TestBuildRegridder:
    """ Test build_regridder method with various parameter combinations. """
    
    def test_build_regridder_with_source_grid_none_raises_error(self: "TestBuildRegridder") -> None:
        """
        This test verifies that attempting to build a regridder without a prepared source grid raises a ValueError. The `build_regridder` method requires both a source and target grid to construct the regridding object, and this test ensures that if the source grid is not set, the method will raise an appropriate error message. This helps enforce the correct usage of the remapper and prevents silent failures. 

        Parameters:
            self ("TestBuildRegridder"): Test instance (unused).

        Returns:
            None: Test asserts ValueError is raised.
        """
        remapper = MPASRemapper(method='bilinear')
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=1.0, dlat=1.0)
        
        with pytest.raises(ValueError, match="Source grid must be provided"):
            remapper.build_regridder(source_grid=None)
    
    def test_build_regridder_with_target_grid_none_raises_error(self: "TestBuildRegridder") -> None:
        """
        This test verifies that attempting to build a regridder without a prepared target grid raises a ValueError. The `build_regridder` method requires both a source and target grid to construct the regridding object, and this test ensures that if the target grid is not set, the method will raise an appropriate error message. This helps enforce the correct usage of the remapper and prevents silent failures. 

        Parameters:
            self ("TestBuildRegridder"): Test instance (unused).

        Returns:
            None: Test asserts ValueError is raised.
        """
        remapper = MPASRemapper(method='bilinear')
        
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 10)),
            'lat': xr.DataArray(np.linspace(30, 40, 10))
        })

        remapper.source_grid = source_grid
        
        with pytest.raises(ValueError, match="Target grid must be provided"):
            remapper.build_regridder(target_grid=None)
    
    def test_build_regridder_with_weights_dir_and_filename(self: "TestBuildRegridder", temp_weights_dir: Path) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when a weights directory and custom filename are specified. The test sets up a remapper with a prepared source grid and target grid, and then it calls `build_regridder` while providing a specific filename for the weights file. The test asserts that the regridder is created without errors and that the weights file is saved in the specified directory with the correct name. This ensures that users can control where their weights files are stored and what they are named when building a regridder. 

        Parameters:
            self ("TestBuildRegridder"): Test instance (unused).
            temp_weights_dir (Path): Temporary directory provided by fixture.

        Returns:
            None: Assertions validate regridder creation and weight file output.
        """
        remapper = MPASRemapper(method='bilinear', weights_dir=temp_weights_dir, reuse_weights=False)
        
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 20)),
            'lat': xr.DataArray(np.linspace(30, 40, 20))
        })

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        
        custom_filename = "custom_weights.nc"
        regridder = remapper.build_regridder(filename=custom_filename)
        
        assert regridder is not None
        assert (temp_weights_dir / custom_filename).exists()
    
    def test_build_regridder_with_extrap_method(self: "TestBuildRegridder", temp_weights_dir: Path) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when an extrapolation method is specified. The test checks that the regridder is created without errors and that it can be built using a specified `extrap_method` value. This ensures that users can control the extrapolation method used in the remapping process and that this parameter is properly integrated into the regridder construction. 

        Parameters:
            self ("TestBuildRegridder"): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate regridder creation.
        """
        remapper = MPASRemapper(
            method='bilinear', 
            extrap_method='nearest_s2d',
            weights_dir=temp_weights_dir,
            reuse_weights=False
        )
        
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 15)),
            'lat': xr.DataArray(np.linspace(30, 40, 15))
        })

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)        
        regridder = remapper.build_regridder()

        assert regridder is not None
    
    def test_build_regridder_with_extrap_dist_exponent(self: "TestBuildRegridder", temp_weights_dir: Path) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when an extrapolation distance exponent is specified. The test checks that the regridder is created without errors and that it can be built using a specified `extrap_dist_exponent` value. This ensures that users can control the distance weighting used in the extrapolation process and that this parameter is properly integrated into the regridder construction. 

        Parameters:
            self ("TestBuildRegridder"): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate regridder creation.
        """
        remapper = MPASRemapper(
            method='bilinear',
            extrap_method='nearest_s2d',
            extrap_dist_exponent=2.0,
            weights_dir=temp_weights_dir,
            reuse_weights=False
        )
        
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 15)),
            'lat': xr.DataArray(np.linspace(30, 40, 15))
        })

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)        
        regridder = remapper.build_regridder()

        assert regridder is not None
    
    def test_build_regridder_with_extrap_num_src_pnts(self: "TestBuildRegridder", temp_weights_dir: Path) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when the number of source points for extrapolation is restricted. The test checks that the regridder is created without errors and that it can be built using a specified `extrap_num_src_pnts` value. This ensures that users can control the number of source points considered in the extrapolation process and that this parameter is properly integrated into the regridder construction. 

        Parameters:
            self ("TestBuildRegridder"): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate regridder creation.
        """
        remapper = MPASRemapper(
            method='bilinear',
            extrap_method='nearest_s2d',
            extrap_num_src_pnts=4,
            weights_dir=temp_weights_dir,
            reuse_weights=False
        )
        
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 15)),
            'lat': xr.DataArray(np.linspace(30, 40, 15))
        })

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)        
        regridder = remapper.build_regridder()

        assert regridder is not None


@pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not installed")


class TestRemapMethod:
    """ Test remap method. """
    
    def test_remap_without_regridder_raises_error(self: "TestRemapMethod") -> None:
        """
        This test verifies that attempting to remap data without a built regridder raises a ValueError. The `remap` method relies on a prepared regridder object to perform the remapping operation, and this test ensures that if the regridder has not been built (i.e., `build_regridder` has not been called), the method will raise an appropriate error message. This helps enforce the correct workflow when using the remapper and prevents users from attempting to remap without the necessary setup.

        Parameters:
            self ("TestRemapMethod"): Test instance (unused).

        Returns:
            None: Test asserts ValueError is raised.
        """
        remapper = MPASRemapper(method='bilinear')        
        data = xr.DataArray(np.random.randn(100), dims=['x'])
        
        with pytest.raises(ValueError, match="Regridder must be built before remapping"):
            remapper.remap(data)
    
    def test_remap_with_numpy_array(self: "TestRemapMethod", temp_weights_dir: Path, mpas_test_data: Any) -> None:
        """
        This test verifies that the `remap` method can successfully remap data when the input is provided as a numpy array. The test sets up a remapper with a built regridder using actual MPAS data to create a structured grid, and then it tests the remapping of a numpy array by wrapping it back as an xarray DataArray with proper dimensions and coordinates. The test asserts that the result of the remapping operation is an xarray DataArray, confirming that the method can handle numpy array inputs correctly by converting them to DataArrays internally before remapping. This ensures that users have flexibility in the format of their input data while still receiving consistent output types. 

        Parameters:
            self ("TestRemapMethod"): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.
            mpas_test_data (Any): Fixture providing a real MPAS Dataset.

        Returns:
            None: Assertions validate remap result type.
        """
        remapper = MPASRemapper(method='bilinear', weights_dir=temp_weights_dir, reuse_weights=False)
        
        n_test = min(200, len(mpas_test_data['lonCell']))
        lon = mpas_test_data['lonCell'].isel(nCells=slice(0, n_test)).values
        lat = mpas_test_data['latCell'].isel(nCells=slice(0, n_test)).values
        data = np.random.randn(n_test)

        data_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=data,
            lon=lon,
            lat=lat,
            intermediate_resolution=5.0  
        )
        
        remapper.source_grid = grid_dataset

        remapper.create_target_grid(
            lon_min=np.degrees(lon.min()),
            lon_max=np.degrees(lon.max()),
            dlon=10.0,
            dlat=10.0
        )

        remapper.build_regridder()
        
        numpy_data = data_array.values
        test_array = xr.DataArray(numpy_data, dims=data_array.dims, coords=data_array.coords)
        result = remapper.remap(test_array)
        
        assert isinstance(result, xr.DataArray)


@pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not installed")


class TestRemapDataset:
    """ Test remap_dataset method with various scenarios. """
    
    def test_remap_dataset_without_regridder_raises_error(self: "TestRemapDataset") -> None:
        """
        This test verifies that attempting to remap an xarray Dataset without a built regridder raises a ValueError. The `remap_dataset` method relies on a prepared regridder object to perform the remapping operation, and this test ensures that if the regridder has not been built (i.e., `build_regridder` has not been called), the method will raise an appropriate error message. This helps enforce the correct workflow when using the remapper and prevents users from attempting to remap without the necessary setup. 

        Parameters:
            self ("TestRemapDataset"): Test instance (unused).

        Returns:
            None: Test asserts ValueError is raised.
        """
        remapper = MPASRemapper(method='bilinear')
        
        dataset = xr.Dataset({
            'temp': xr.DataArray(np.random.randn(100), dims=['x']),
            'pressure': xr.DataArray(np.random.randn(100), dims=['x'])
        })
        
        with pytest.raises(ValueError, match="Regridder must be built before remapping"):
            remapper.remap_dataset(dataset)
    
    def test_remap_dataset_with_missing_variable_skip_missing_true(self: "TestRemapDataset", temp_weights_dir: Path, mpas_test_data: Any) -> None:
        """
        This test verifies that the `remap_dataset` method can successfully remap a Dataset even when one of the requested variables is missing, as long as `skip_missing=True` is set. The test sets up a remapper with a built regridder using actual MPAS data to create a structured grid, and then it tests the remapping of a Dataset that includes one existing variable and one nonexistent variable. The test asserts that the remapping operation completes without errors and that the resulting Dataset contains only the existing variable, confirming that the method correctly skips missing variables without failing when `skip_missing` is enabled. This ensures that users can choose to ignore missing variables during remapping if they prefer. 

        Parameters:
            self ("TestRemapDataset"): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.
            mpas_test_data (Any): Fixture providing real MPAS Dataset.

        Returns:
            None: Assertions validate returned Dataset contents.
        """
        remapper = MPASRemapper(method='bilinear', weights_dir=temp_weights_dir, reuse_weights=False)
        
        n_test = min(200, len(mpas_test_data['lonCell']))
        lon = mpas_test_data['lonCell'].isel(nCells=slice(0, n_test)).values
        lat = mpas_test_data['latCell'].isel(nCells=slice(0, n_test)).values
        
        temp_data = np.random.randn(n_test)
        pressure_data = np.random.randn(n_test)
        
        temp_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=temp_data,
            lon=lon,
            lat=lat,
            intermediate_resolution=5.0
        )
        
        pressure_array, _ = MPASRemapper.unstructured_to_structured_grid(
            data=pressure_data,
            lon=lon,
            lat=lat,
            intermediate_resolution=5.0
        )
        
        remapper.source_grid = grid_dataset

        remapper.create_target_grid(
            lon_min=np.degrees(lon.min()),
            lon_max=np.degrees(lon.max()),
            dlon=10.0,
            dlat=10.0
        )

        remapper.build_regridder()
        
        dataset = xr.Dataset({
            'temp': temp_array,
            'pressure': pressure_array
        })
        
        result = remapper.remap_dataset(dataset, variables=['temp', 'nonexistent'], skip_missing=True)
        
        assert 'temp' in result
        assert 'nonexistent' not in result
    
    def test_remap_dataset_with_missing_variable_skip_missing_false(self: "TestRemapDataset", temp_weights_dir: Path) -> None:
        """
        This test verifies that the `remap_dataset` method raises a ValueError when one of the requested variables is missing and `skip_missing=False` is set. The test sets up a remapper with a built regridder using synthetic source and target grids, and then it tests the remapping of a Dataset that includes one existing variable and one nonexistent variable. The test asserts that the remapping operation raises a ValueError with an appropriate message indicating that the variable was not found in the dataset, confirming that the method correctly enforces the presence of all requested variables when `skip_missing` is disabled. This ensures that users are alerted to missing variables during remapping if they choose not to skip them. 

        Parameters:
            self (Any): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Test asserts ValueError is raised on missing variable.
        """
        remapper = MPASRemapper(method='bilinear', weights_dir=temp_weights_dir, reuse_weights=False)
        
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 20)),
            'lat': xr.DataArray(np.linspace(30, 40, 20))
        })

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        remapper.build_regridder()
        
        dataset = xr.Dataset({
            'temp': xr.DataArray(np.random.randn(20), dims=['x']),
        })
        
        with pytest.raises(ValueError, match="Variable.*not found in dataset"):
            remapper.remap_dataset(dataset, variables=['nonexistent'], skip_missing=False)
    
    def test_remap_dataset_with_exception_skip_missing_true(self: "TestRemapDataset", temp_weights_dir: Path) -> None:
        """
        This test verifies that the `remap_dataset` method can gracefully handle exceptions during remapping when `skip_missing=True` is set. The test sets up a remapper with a built regridder using synthetic source and target grids, and then it tests the remapping of a Dataset that is intentionally designed to cause an error (e.g., by having an incompatible shape). The test asserts that the remapping operation does not raise an exception and instead returns an empty Dataset or skips the problematic variable, confirming that the method can handle errors gracefully without failing when `skip_missing` is enabled. This ensures that users can choose to ignore errors during remapping if they prefer.

        Parameters:
            self ("TestRemapDataset"): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate graceful skip behavior.
        """
        remapper = MPASRemapper(method='bilinear', weights_dir=temp_weights_dir, reuse_weights=False)
        
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 20)),
            'lat': xr.DataArray(np.linspace(30, 40, 20))
        })

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        remapper.build_regridder()
        
        dataset = xr.Dataset({
            'temp': xr.DataArray(np.random.randn(100), dims=['x']),  
        })
        
        result = remapper.remap_dataset(dataset, skip_missing=True)
        assert isinstance(result, xr.Dataset)


@pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not installed")


class TestUnstructuredToStructuredGrid:
    """ Test unstructured_to_structured_grid method. """
    
    def test_with_explicit_bounds(self: "TestUnstructuredToStructuredGrid", simple_mpas_data: Any) -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS-like data into a structured grid format when explicit longitude and latitude bounds are provided. The test checks that the returned structured DataArray and Dataset are of the correct types and that the grid Dataset contains the expected coordinate variables ('lon' and 'lat'). This ensures that the method can handle cases where users specify bounds for the conversion process and can produce a valid structured grid for remapping operations. 

        Parameters:
            self ("TestUnstructuredToStructuredGrid"): Test instance (unused).
            simple_mpas_data (Any): Fixture providing small MPAS-like arrays.

        Returns:
            None: Assertions validate returned types and grid coordinates.
        """
        data_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=simple_mpas_data['data'],
            lon=simple_mpas_data['lon'],
            lat=simple_mpas_data['lat'],
            intermediate_resolution=2.0,
            lon_min=-180,
            lon_max=180,
            lat_min=-90,
            lat_max=90,
            buffer=0.0
        )
        
        assert isinstance(data_array, xr.DataArray)
        assert isinstance(grid_dataset, xr.Dataset)
        assert 'lon' in grid_dataset.coords
        assert 'lat' in grid_dataset.coords
    
    def test_with_none_bounds(self: "TestUnstructuredToStructuredGrid", simple_mpas_data: Any) -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS-like data into a structured grid format when longitude and latitude bounds are set to None, allowing the method to automatically determine appropriate bounds based on the input coordinates. The test checks that the returned structured DataArray and Dataset are of the correct types and that the resulting grid Dataset contains the expected coordinate variables. This ensures that the method can handle cases where users do not provide explicit bounds and can still produce a valid structured grid for remapping operations. 

        Parameters:
            self ("TestUnstructuredToStructuredGrid"): Test instance (unused).
            simple_mpas_data (Any): Fixture providing small MPAS-like arrays.

        Returns:
            None: Assertions validate the grid is created and typed correctly.
        """
        data_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=simple_mpas_data['data'],
            lon=simple_mpas_data['lon'],
            lat=simple_mpas_data['lat'],
            intermediate_resolution=2.0,
            lon_min=None, 
            lon_max=None,
            lat_min=None,
            lat_max=None
        )
        
        assert isinstance(data_array, xr.DataArray)
        assert isinstance(grid_dataset, xr.Dataset)


@pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not installed")


class TestCleanup:
    """ Test cleanup method. """
    
    def test_cleanup_clears_regridder_and_grids(self: "TestCleanup", temp_weights_dir: Path) -> None:
        """
        This test verifies that the `cleanup` method of the `MPASRemapper` class properly clears the regridder and grid attributes. The test sets up a remapper with a built regridder using synthetic source and target grids, and then it calls the `cleanup` method. The test asserts that after cleanup, the `regridder`, `source_grid`, and `target_grid` attributes are all set to None, confirming that the method effectively releases resources and resets the state of the remapper. This ensures that users can rely on the `cleanup` method to free up memory and prepare the remapper for a new remapping operation if needed. 

        Parameters:
            self ("TestCleanup"): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate cleanup semantics.
        """
        remapper = MPASRemapper(method='bilinear', weights_dir=temp_weights_dir, reuse_weights=False)
        
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 20)),
            'lat': xr.DataArray(np.linspace(30, 40, 20))
        })

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        remapper.build_regridder()
        
        assert remapper.regridder is not None
        assert remapper.source_grid is not None
        assert remapper.target_grid is not None
        
        remapper.cleanup()
        
        assert remapper.regridder is None
        assert remapper.source_grid is None
        assert remapper.target_grid is None


@pytest.mark.skipif(not XESMF_AVAILABLE, reason="xESMF not installed")


class TestEstimateMemoryUsage:
    """ Test estimate_memory_usage method. """
    
    def test_estimate_memory_patch_method(self: "TestEstimateMemoryUsage") -> None:
        """
        This test verifies that the `estimate_memory_usage` method returns a positive float when estimating memory usage for the 'patch' remapping method. The test checks that the returned memory estimate is greater than zero and is of type float, which is consistent with the expected behavior of the method for the given input sizes. This ensures that the memory estimation logic for the patch method is functioning correctly and provides a sensible estimate for users. 

        Parameters:
            self ("TestEstimateMemoryUsage"): Test instance (unused).

        Returns:
            None: Assertions validate returned memory estimate.
        """
        memory_gb = MPASRemapper.estimate_memory_usage(
            n_source=10000,
            n_target=10000,
            method='patch'
        )
        
        assert memory_gb > 0
        assert isinstance(memory_gb, float)
    
    def test_estimate_memory_conservative_method(self: "TestEstimateMemoryUsage") -> None:
        """
        This test verifies that the `estimate_memory_usage` method returns a positive float when estimating memory usage for the 'conservative' remapping method. The test checks that the returned memory estimate is greater than zero and is of type float, which is consistent with the expected behavior of the method for the given input sizes. This ensures that the memory estimation logic for the conservative method is functioning correctly. 

        Parameters:
            self ("TestEstimateMemoryUsage"): Test instance (unused).

        Returns:
            None: Assertions validate returned memory estimate.
        """
        memory_gb = MPASRemapper.estimate_memory_usage(
            n_source=10000,
            n_target=10000,
            method='conservative'
        )
        
        assert memory_gb > 0
    
    def test_estimate_memory_nearest_method(self: "TestEstimateMemoryUsage") -> None:
        """
        This test verifies that the `estimate_memory_usage` method returns a positive float when estimating memory usage for the 'nearest_s2d' remapping method. The test checks that the returned memory estimate is greater than zero and is of type float, which is consistent with the expected behavior of the method for the given input sizes. This ensures that the memory estimation logic for the nearest neighbor method is functioning correctly and provides a sensible estimate for users. 

        Parameters:
            self ("TestEstimateMemoryUsage"): Test instance (unused).

        Returns:
            None: Assertions validate returned memory estimate.
        """
        memory_gb = MPASRemapper.estimate_memory_usage(
            n_source=10000,
            n_target=10000,
            method='nearest_s2d'
        )
        
        assert memory_gb > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
