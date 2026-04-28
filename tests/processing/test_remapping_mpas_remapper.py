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
import xarray as xr
from typing import Any
from pathlib import Path

from mpasdiag.processing.remapping import (
    MPASRemapper,
    ESMPY_AVAILABLE,
)

from tests.test_data_helpers import load_mpas_coords_from_processor


class TestMPASRemapper:
    """ Test the MPASRemapper class """
    
    def setup_method(self: 'TestMPASRemapper') -> None:
        """
        This setup method prepares the test environment for each test in the `TestMPASRemapper` class. It creates a temporary directory to store any weights files generated during the tests and loads synthetic MPAS coordinate data (longitude, latitude, and associated data) using a helper function. The longitude values are normalized to the 0-360 degree range to ensure consistency with typical MPAS grid conventions. The prepared coordinates and data are stored as instance attributes for use in the various remapping tests that follow.

        Parameters:
            self ('TestMPASRemapper'): Test instance to receive prepared attributes.

        Returns:
            None: Populates instance attributes for use by tests.
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.n_cells = 300
        lon, lat, u, v = load_mpas_coords_from_processor(n=self.n_cells)
        self.mpas_lon = np.mod(lon, 360.0)
        self.mpas_lat = lat
        self.mpas_data = 20 + 10 * (u - u.min()) / (u.max() - u.min() + 1e-12)
    
    def teardown_method(self: 'TestMPASRemapper') -> None:
        """
        This teardown method cleans up the test environment after each test in the `TestMPASRemapper` class. It checks if the temporary directory created during setup exists and, if so, it removes the directory and all of its contents using `shutil.rmtree`. This ensures that any weights files or other temporary data generated during the tests do not persist on the filesystem, maintaining a clean state for subsequent tests and preventing clutter.

        Parameters:
            self ('TestMPASRemapper'): Test instance (unused).

        Returns:
            None: Performs cleanup operations.
        """
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    
    def test_build_regridder_error_no_target(self: 'TestMPASRemapper') -> None:
        """
        This test verifies that attempting to build a regridder without a prepared target grid raises a ValueError. The `build_regridder` method requires both a source and target grid to construct the regridding object, and this test ensures that if the target grid is not set, the method will raise an appropriate error message. This helps enforce the correct usage of the remapper and prevents silent failures. 

        Parameters:
            self ('TestMPASRemapper'): Test instance (unused).

        Returns:
            None: Test asserts the expected ValueError is raised.
        """
        remapper = MPASRemapper()
        remapper.prepare_source_grid(self.mpas_lon, self.mpas_lat)
        
        with pytest.raises(ValueError) as context:
            remapper.build_regridder()

        assert 'Target grid' in str(context.value)
    
    
    def test_cleanup(self: 'TestMPASRemapper') -> None:
        """
        This test verifies that the `cleanup` method of the `MPASRemapper` class properly clears internal attributes related to the source grid, target grid, and regridder. After calling `cleanup`, the test asserts that these attributes are set to `None`, confirming that the method effectively resets the state of the remapper. This is important for ensuring that resources are released and that the remapper can be reused or safely discarded without lingering references to previous grids or regridders. 

        Parameters:
            self ('TestMPASRemapper'): Test instance with a configured remapper.

        Returns:
            None: Assertions validate that internal attributes are cleared.
        """
        remapper = MPASRemapper()
        remapper.prepare_source_grid(self.mpas_lon, self.mpas_lat)
        remapper.create_target_grid()
        
        remapper.cleanup()

        assert remapper.source_grid is None
        assert remapper.target_grid is None
        assert remapper._weights is None
    
    
    def test_unstructured_to_structured_grid(self: 'TestMPASRemapper') -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS grid data into a structured grid format. The method should return a structured DataArray and a corresponding Dataset for the grid, and the test asserts that these outputs are of the correct types and contain the expected dimensions and coordinate variables. This ensures that the conversion process is functioning correctly and that the resulting structured grid is properly formatted for use in remapping operations.

        Parameters:
            self ('TestMPASRemapper'): Test instance with synthetic MPAS data.

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
    
    def test_unstructured_to_structured_grid_auto_bounds(self: 'TestMPASRemapper') -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS grid data into a structured grid format without explicitly providing longitude and latitude bounds. The method should automatically determine appropriate bounds based on the input coordinates and return a structured DataArray and a corresponding Dataset for the grid. The test asserts that the outputs are of the correct types and contain the expected coordinate variables, confirming that the method can handle cases where bounds are not provided and can still produce a valid structured grid for remapping operations. 

        Parameters:
            self ('TestMPASRemapper'): Test instance with synthetic MPAS data.

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
    
    def test_unstructured_to_structured_with_dataarray(self: 'TestMPASRemapper') -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert an xarray DataArray defined on an unstructured MPAS grid into a structured grid format. The method should preserve the attributes of the input DataArray and return a structured DataArray and a corresponding Dataset for the grid. The test asserts that the output DataArray retains the original attributes (e.g., 'units', 'long_name') and that the structured grid is properly formatted for use in remapping operations. This ensures that users can seamlessly convert their unstructured MPAS data into a structured format while maintaining important metadata. 

        Parameters:
            self ('TestMPASRemapper'): Test instance with synthetic MPAS DataArray input.

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


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")


class TestBuildRegridder:
    """ Test build_regridder method with various parameter combinations. """
    
    
    def test_build_regridder_with_weights_dir_and_filename(self: 'TestBuildRegridder', temp_weights_dir: Path) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when a weights directory and custom filename are specified. The test sets up a remapper with a prepared source grid and target grid, and then it calls `build_regridder` while providing a specific filename for the weights file. The test asserts that the regridder is created without errors and that the weights file is saved in the specified directory with the correct name. This ensures that users can control where their weights files are stored and what they are named when building a regridder. 

        Parameters:
            self ('TestBuildRegridder'): Test instance (unused).
            temp_weights_dir (Path): Temporary directory provided by fixture.

        Returns:
            None: Assertions validate regridder creation and weight file output.
        """
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=temp_weights_dir, reuse_weights=False)
        
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
    
    def test_build_regridder_with_extrap_method(self: 'TestBuildRegridder', temp_weights_dir: Path) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when an extrapolation method is specified. The test checks that the regridder is created without errors and that it can be built using a specified `extrap_method` value. This ensures that users can control the extrapolation method used in the remapping process and that this parameter is properly integrated into the regridder construction. 

        Parameters:
            self ('TestBuildRegridder'): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate regridder creation.
        """
        remapper = MPASRemapper(
            method='nearest_s2d', 
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
    
    def test_build_regridder_with_extrap_dist_exponent(self: 'TestBuildRegridder', temp_weights_dir: Path) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when an extrapolation distance exponent is specified. The test checks that the regridder is created without errors and that it can be built using a specified `extrap_dist_exponent` value. This ensures that users can control the distance weighting used in the extrapolation process and that this parameter is properly integrated into the regridder construction. 

        Parameters:
            self ('TestBuildRegridder'): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate regridder creation.
        """
        remapper = MPASRemapper(
            method='nearest_s2d',
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
    
    def test_build_regridder_with_extrap_num_src_pnts(self: 'TestBuildRegridder', temp_weights_dir: Path) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when the number of source points for extrapolation is restricted. The test checks that the regridder is created without errors and that it can be built using a specified `extrap_num_src_pnts` value. This ensures that users can control the number of source points considered in the extrapolation process and that this parameter is properly integrated into the regridder construction. 

        Parameters:
            self ('TestBuildRegridder'): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate regridder creation.
        """
        remapper = MPASRemapper(
            method='nearest_s2d',
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


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")


class TestRemapMethod:
    """ Test remap method. """
    
    
    def test_remap_with_numpy_array(self: 'TestRemapMethod', temp_weights_dir: Path, mpas_test_data: Any) -> None:
        """
        This test verifies that the `remap` method can successfully remap data when the input is provided as a numpy array. The test sets up a remapper with a built regridder using actual MPAS data to create a structured grid, and then it tests the remapping of a numpy array by wrapping it back as an xarray DataArray with proper dimensions and coordinates. The test asserts that the result of the remapping operation is an xarray DataArray, confirming that the method can handle numpy array inputs correctly by converting them to DataArrays internally before remapping. This ensures that users have flexibility in the format of their input data while still receiving consistent output types. 

        Parameters:
            self ('TestRemapMethod'): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.
            mpas_test_data (Any): Fixture providing a real MPAS Dataset.

        Returns:
            None: Assertions validate remap result type.
        """
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=temp_weights_dir, reuse_weights=False)
        
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


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")


class TestRemapDataset:
    """ Test remap_dataset method with various scenarios. """
    
    
    def test_remap_dataset_with_missing_variable_skip_missing_true(self: 'TestRemapDataset', temp_weights_dir: Path, mpas_test_data: Any) -> None:
        """
        This test verifies that the `remap_dataset` method can successfully remap a Dataset even when one of the requested variables is missing, as long as `skip_missing=True` is set. The test sets up a remapper with a built regridder using actual MPAS data to create a structured grid, and then it tests the remapping of a Dataset that includes one existing variable and one nonexistent variable. The test asserts that the remapping operation completes without errors and that the resulting Dataset contains only the existing variable, confirming that the method correctly skips missing variables without failing when `skip_missing` is enabled. This ensures that users can choose to ignore missing variables during remapping if they prefer. 

        Parameters:
            self ('TestRemapDataset'): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.
            mpas_test_data (Any): Fixture providing real MPAS Dataset.

        Returns:
            None: Assertions validate returned Dataset contents.
        """
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=temp_weights_dir, reuse_weights=False)
        
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
    
    def test_remap_dataset_with_missing_variable_skip_missing_false(self: 'TestRemapDataset', temp_weights_dir: Path) -> None:
        """
        This test verifies that the `remap_dataset` method raises a ValueError when one of the requested variables is missing and `skip_missing=False` is set. The test sets up a remapper with a built regridder using synthetic source and target grids, and then it tests the remapping of a Dataset that includes one existing variable and one nonexistent variable. The test asserts that the remapping operation raises a ValueError with an appropriate message indicating that the variable was not found in the dataset, confirming that the method correctly enforces the presence of all requested variables when `skip_missing` is disabled. This ensures that users are alerted to missing variables during remapping if they choose not to skip them. 

        Parameters:
            self (Any): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Test asserts ValueError is raised on missing variable.
        """
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=temp_weights_dir, reuse_weights=False)
        
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
    
    def test_remap_dataset_with_exception_skip_missing_true(self: 'TestRemapDataset', temp_weights_dir: Path) -> None:
        """
        This test verifies that the `remap_dataset` method can gracefully handle exceptions during remapping when `skip_missing=True` is set. The test sets up a remapper with a built regridder using synthetic source and target grids, and then it tests the remapping of a Dataset that is intentionally designed to cause an error (e.g., by having an incompatible shape). The test asserts that the remapping operation does not raise an exception and instead returns an empty Dataset or skips the problematic variable, confirming that the method can handle errors gracefully without failing when `skip_missing` is enabled. This ensures that users can choose to ignore errors during remapping if they prefer.

        Parameters:
            self ('TestRemapDataset'): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate graceful skip behavior.
        """
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=temp_weights_dir, reuse_weights=False)
        
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


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")


class TestUnstructuredToStructuredGrid:
    """ Test unstructured_to_structured_grid method. """
    
    def test_with_explicit_bounds(self: 'TestUnstructuredToStructuredGrid', simple_mpas_data: Any) -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS-like data into a structured grid format when explicit longitude and latitude bounds are provided. The test checks that the returned structured DataArray and Dataset are of the correct types and that the grid Dataset contains the expected coordinate variables ('lon' and 'lat'). This ensures that the method can handle cases where users specify bounds for the conversion process and can produce a valid structured grid for remapping operations. 

        Parameters:
            self ('TestUnstructuredToStructuredGrid'): Test instance (unused).
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
    
    def test_with_none_bounds(self: 'TestUnstructuredToStructuredGrid', simple_mpas_data: Any) -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS-like data into a structured grid format when longitude and latitude bounds are set to None, allowing the method to automatically determine appropriate bounds based on the input coordinates. The test checks that the returned structured DataArray and Dataset are of the correct types and that the resulting grid Dataset contains the expected coordinate variables. This ensures that the method can handle cases where users do not provide explicit bounds and can still produce a valid structured grid for remapping operations. 

        Parameters:
            self ('TestUnstructuredToStructuredGrid'): Test instance (unused).
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


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")


class TestCleanup:
    """ Test cleanup method. """
    
    def test_cleanup_clears_regridder_and_grids(self: 'TestCleanup', temp_weights_dir: Path) -> None:
        """
        This test verifies that the `cleanup` method of the `MPASRemapper` class properly clears the regridder and grid attributes. The test sets up a remapper with a built regridder using synthetic source and target grids, and then it calls the `cleanup` method. The test asserts that after cleanup, the `regridder`, `source_grid`, and `target_grid` attributes are all set to None, confirming that the method effectively releases resources and resets the state of the remapper. This ensures that users can rely on the `cleanup` method to free up memory and prepare the remapper for a new remapping operation if needed. 

        Parameters:
            self ('TestCleanup'): Test instance (unused).
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None: Assertions validate cleanup semantics.
        """
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=temp_weights_dir, reuse_weights=False)
        
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 20)),
            'lat': xr.DataArray(np.linspace(30, 40, 20))
        })

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        remapper.build_regridder()

        assert remapper._weights is not None
        assert remapper.source_grid is not None
        assert remapper.target_grid is not None

        remapper.cleanup()

        assert remapper._weights is None
        assert remapper.source_grid is None
        assert remapper.target_grid is None


class TestRemapDatasetSkipMissingFalse:
    """Tests for remap_dataset with skip_missing=False raising on error (line 658)."""

    @pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")
    def test_remap_dataset_skip_missing_false_raises(self: 'TestRemapDatasetSkipMissingFalse', tmp_path: Path) -> None:
        """When skip_missing=False and a variable fails, remap_dataset must re-raise."""
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=tmp_path,
                                reuse_weights=False)
        source_grid = xr.Dataset({
            'lon': xr.DataArray(np.linspace(-120, -110, 20)),
            'lat': xr.DataArray(np.linspace(30, 40, 20)),
        })
        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        remapper.build_regridder()

        # Variable with a shape incompatible with the source grid triggers an error
        bad_dataset = xr.Dataset({
            'bad_var': xr.DataArray(np.random.randn(100), dims=['x']),
        })
        with pytest.raises(Exception):
            remapper.remap_dataset(bad_dataset, skip_missing=False)


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")
class TestMPASRemapperCoveragePaths:
    """Tests targeting ESMPy code paths not hit by existing tests."""

    # ------------------------------------------------------------------ #
    #  Shared fixtures                                                     #
    # ------------------------------------------------------------------ #

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.fixture
    def unstructured_source(self: 'TestMPASRemapperCoveragePaths') -> xr.Dataset:
        """Minimal unstructured (1-D lon/lat) source grid."""
        n = 60
        rng = np.random.default_rng(77)
        lon = rng.uniform(-110, -100, n)
        lat = rng.uniform(30, 40, n)
        return xr.Dataset({
            'lon': xr.DataArray(lon, dims=['nCells']),
            'lat': xr.DataArray(lat, dims=['nCells']),
        })

    @pytest.fixture
    def structured_source(self: 'TestMPASRemapperCoveragePaths') -> xr.Dataset:
        """Structured 2-D (lon has 'lon' dim, lat has 'lat' dim) source grid."""
        lon = np.linspace(-110, -100, 12)
        lat = np.linspace(30, 40, 10)
        return xr.Dataset({
            'lon': xr.DataArray(lon, dims=['lon']),
            'lat': xr.DataArray(lat, dims=['lat']),
        })

    def _build_nearest_remapper(self: 'TestMPASRemapperCoveragePaths', source, temp_dir, reuse=False):
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=temp_dir,
                                reuse_weights=reuse)
        remapper.source_grid = source
        remapper.create_target_grid(lon_min=-110, lon_max=-100,
                                    lat_min=30, lat_max=40,
                                    dlon=2.0, dlat=2.0)
        return remapper

    # ------------------------------------------------------------------ #
    #  Weight caching: _try_load_cached_weights (lines 324-331)          #
    # ------------------------------------------------------------------ #

    def test_weight_caching_reuses_saved_file(self: 'TestMPASRemapperCoveragePaths', unstructured_source, temp_dir) -> None:
        """Second build_regridder call with reuse_weights=True loads from file."""
        fname = "cached_weights.nc"
        # First build: saves weights
        r1 = self._build_nearest_remapper(unstructured_source, temp_dir, reuse=False)
        r1.build_regridder(filename=fname)
        weights_file = temp_dir / fname
        assert weights_file.exists()

        # Second build: same file exists + reuse_weights=True → loads cache
        r2 = MPASRemapper(method='nearest_s2d', weights_dir=temp_dir, reuse_weights=True)
        r2.source_grid = unstructured_source
        r2.create_target_grid(lon_min=-110, lon_max=-100,
                              lat_min=30, lat_max=40,
                              dlon=2.0, dlat=2.0)
        weights = r2.build_regridder(filename=fname)
        assert weights is not None

    # ------------------------------------------------------------------ #
    #  Weight file I/O: _save_weights_netcdf / _load_weights_netcdf      #
    #  (lines 1090, 1119-1138)                                            #
    # ------------------------------------------------------------------ #

    def test_save_and_load_weights_roundtrip(self: 'TestMPASRemapperCoveragePaths', unstructured_source, temp_dir) -> None:
        """build_regridder with weights_dir saves a file loadable by _load_weights_netcdf."""
        remapper = self._build_nearest_remapper(unstructured_source, temp_dir, reuse=False)
        remapper.build_regridder(filename="roundtrip.nc")
        weights_path = temp_dir / "roundtrip.nc"
        assert weights_path.exists()

        weight_matrix, tgt_shape, cell_of_element = MPASRemapper._load_weights_netcdf(
            weights_path
        )
        assert weight_matrix is not None
        assert len(tgt_shape) == 2
        assert tgt_shape[0] > 0 and tgt_shape[1] > 0

    def test_load_weights_with_cell_of_element(self: 'TestMPASRemapperCoveragePaths', unstructured_source, temp_dir) -> None:
        """_load_weights_netcdf returns cell_of_element=None for non-mesh sources."""
        remapper = self._build_nearest_remapper(unstructured_source, temp_dir, reuse=False)
        remapper.build_regridder(filename="coe_test.nc")
        weight_matrix, tgt_shape, coe = MPASRemapper._load_weights_netcdf(
            temp_dir / "coe_test.nc"
        )
        # Unstructured locstream source has no cell_of_element
        assert coe is None

    # ------------------------------------------------------------------ #
    #  patch method raises ValueError (lines 493-504)                     #
    # ------------------------------------------------------------------ #

    def test_patch_method_raises_value_error(self: 'TestMPASRemapperCoveragePaths', unstructured_source, temp_dir) -> None:
        """_build_weights_on_rank0 raises ValueError for patch method."""
        remapper = MPASRemapper(method='patch', weights_dir=temp_dir, reuse_weights=False)
        remapper.source_grid = unstructured_source
        remapper.create_target_grid(lon_min=-110, lon_max=-100,
                                    lat_min=30, lat_max=40,
                                    dlon=2.0, dlat=2.0)
        with pytest.raises(ValueError, match="patch"):
            remapper.build_regridder()

    # ------------------------------------------------------------------ #
    #  Conservative path: missing boundaries raises ValueError             #
    #  (lines 352-361)                                                     #
    # ------------------------------------------------------------------ #

    def test_conservative_without_boundaries_raises(self: 'TestMPASRemapperCoveragePaths', unstructured_source,
                                                     temp_dir) -> None:
        """_prepare_source_esmpy raises ValueError when lon_b/lat_b absent."""
        remapper = MPASRemapper(method='conservative', weights_dir=temp_dir,
                                reuse_weights=False)
        remapper.source_grid = unstructured_source  # no lon_b / lat_b
        remapper.create_target_grid(lon_min=-110, lon_max=-100,
                                    lat_min=30, lat_max=40,
                                    dlon=2.0, dlat=2.0)
        with pytest.raises(ValueError, match="[Cc]onservative"):
            remapper.build_regridder()

    # ------------------------------------------------------------------ #
    #  Structured source path (line 416)                                   #
    # ------------------------------------------------------------------ #

    def test_structured_source_builds_successfully(self: 'TestMPASRemapperCoveragePaths', structured_source,
                                                    temp_dir) -> None:
        """Structured (2-D dims) source grid takes the structured path in _prepare_source_esmpy."""
        remapper = MPASRemapper(method='nearest_s2d', weights_dir=temp_dir,
                                reuse_weights=False)
        remapper.source_grid = structured_source
        remapper.create_target_grid(lon_min=-110, lon_max=-100,
                                    lat_min=30, lat_max=40,
                                    dlon=2.0, dlat=2.0)
        weights = remapper.build_regridder()
        assert weights is not None

    # ------------------------------------------------------------------ #
    #  _build_esmpy_grid with add_corners=True (lines 697-698, 950-960)  #
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    #  Conservative remapping with triangular mesh                        #
    #  Covers: lines 359-377, 854-920, 1013, 1090 (cell_of_element save) #
    # ------------------------------------------------------------------ #

    @pytest.fixture
    def triangular_source(self: 'TestMPASRemapperCoveragePaths') -> xr.Dataset:
        """Minimal 4-cell triangular mesh with lon_b/lat_b boundaries."""
        lon_c = np.array([-104.0, -104.0, -101.0, -101.0])
        lat_c = np.array([36.0,   38.0,   36.0,   38.0])
        # Triangular vertices (3 per cell), in degrees
        lon_b = np.array([
            [-105.0, -103.0, -104.0],
            [-105.0, -103.0, -104.0],
            [-102.0, -100.0, -101.0],
            [-102.0, -100.0, -101.0],
        ])
        lat_b = np.array([
            [35.0, 35.0, 37.0],
            [37.0, 37.0, 39.0],
            [35.0, 35.0, 37.0],
            [37.0, 37.0, 39.0],
        ])
        return xr.Dataset({
            'lon': xr.DataArray(lon_c, dims=['nCells']),
            'lat': xr.DataArray(lat_c, dims=['nCells']),
            'lon_b': xr.DataArray(lon_b, dims=['nCells', 'nv']),
            'lat_b': xr.DataArray(lat_b, dims=['nCells', 'nv']),
        })

    def test_conservative_with_triangular_mesh(self: 'TestMPASRemapperCoveragePaths', triangular_source,
                                               temp_dir) -> None:
        """Conservative remapping with actual mesh boundaries builds and remaps."""
        remapper = MPASRemapper(method='conservative', weights_dir=temp_dir,
                                reuse_weights=False)
        remapper.source_grid = triangular_source
        remapper.create_target_grid(lon_min=-106, lon_max=-99,
                                    lat_min=34, lat_max=41,
                                    dlon=2.0, dlat=2.0)
        weights = remapper.build_regridder()
        assert weights is not None

        data = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0]), dims=['nCells'])
        result = remapper.remap(data)
        assert isinstance(result, xr.DataArray)

    def test_conservative_normed_uses_fracarea_norm(self: 'TestMPASRemapperCoveragePaths', triangular_source,
                                                      temp_dir) -> None:
        """conservative_normed method sets norm_type=FRACAREA (line 1044)."""
        remapper = MPASRemapper(method='conservative_normed', weights_dir=temp_dir,
                                reuse_weights=False)
        remapper.source_grid = triangular_source
        remapper.create_target_grid(lon_min=-106, lon_max=-99,
                                    lat_min=34, lat_max=41,
                                    dlon=2.0, dlat=2.0)
        weights = remapper.build_regridder()
        assert weights is not None

    def test_conservative_saves_cell_of_element_in_weights_file(
            self: 'TestMPASRemapperCoveragePaths', triangular_source, temp_dir) -> None:
        """Weight file for conservative remapping includes cell_of_element (line 1090)."""
        remapper = MPASRemapper(method='conservative', weights_dir=temp_dir,
                                reuse_weights=False)
        remapper.source_grid = triangular_source
        remapper.create_target_grid(lon_min=-106, lon_max=-99,
                                    lat_min=34, lat_max=41,
                                    dlon=2.0, dlat=2.0)
        remapper.build_regridder(filename="conservative_weights.nc")
        weights_file = temp_dir / "conservative_weights.nc"
        assert weights_file.exists()

        _, _, cell_of_element = MPASRemapper._load_weights_netcdf(weights_file)
        assert cell_of_element is not None  # mesh sources have cell_of_element


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
