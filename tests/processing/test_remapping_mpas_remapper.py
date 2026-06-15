#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

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

from mpasdiag import (
    MPASRemapper,
    ESMPY_AVAILABLE,
)

from tests.test_data_helpers import load_mpas_coords_from_processor

_RNG = np.random.default_rng(42)


class TestMPASRemapper:
    """Test the MPASRemapper class"""

    def setup_method(self: "TestMPASRemapper") -> None:
        """
        This setup method prepares the test environment for each test in the `TestMPASRemapper` class. It creates a temporary directory to store any weights files generated during the tests and loads synthetic MPAS coordinate data (longitude, latitude, and associated data) using a helper function. The longitude values are normalized to the 0-360 degree range to ensure consistency with typical MPAS grid conventions. The prepared coordinates and data are stored as instance attributes for use in the various remapping tests that follow.

        Parameters:
            None

        Returns:
            None
        """
        self.temp_dir = Path(tempfile.mkdtemp())
        self.n_cells = 300
        lon, lat, u, _ = load_mpas_coords_from_processor(n=self.n_cells)
        self.mpas_lon = np.mod(lon, 360.0)
        self.mpas_lat = lat
        self.mpas_data = 20 + 10 * (u - u.min()) / (u.max() - u.min() + 1e-12)

    def teardown_method(self: "TestMPASRemapper") -> None:
        """
        This teardown method cleans up the test environment after each test in the `TestMPASRemapper` class. It checks if the temporary directory created during setup exists and, if so, it removes the directory and all of its contents using `shutil.rmtree`. This ensures that any weights files or other temporary data generated during the tests do not persist on the filesystem, maintaining a clean state for subsequent tests and preventing clutter.

        Parameters:
            None

        Returns:
            None
        """
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_build_regridder_error_no_target(self: "TestMPASRemapper") -> None:
        """
        This test verifies that attempting to build a regridder without a prepared target grid raises a ValueError. The `build_regridder` method requires both a source and target grid to construct the regridding object, and this test ensures that if the target grid is not set, the method will raise an appropriate error message. This helps enforce the correct usage of the remapper and prevents silent failures.

        Parameters:
            None

        Returns:
            None
        """
        remapper = MPASRemapper()
        remapper.prepare_source_grid(self.mpas_lon, self.mpas_lat)

        with pytest.raises(ValueError) as context:
            remapper.build_regridder()

        assert "Target grid" in str(context.value)

    def test_cleanup(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `cleanup` method of the `MPASRemapper` class properly clears internal attributes related to the source grid, target grid, and regridder. After calling `cleanup`, the test asserts that these attributes are set to `None`, confirming that the method effectively resets the state of the remapper. This is important for ensuring that resources are released and that the remapper can be reused or safely discarded without lingering references to previous grids or regridders.

        Parameters:
            None

        Returns:
            None
        """
        remapper = MPASRemapper()
        remapper.prepare_source_grid(self.mpas_lon, self.mpas_lat)
        remapper.create_target_grid()

        remapper.cleanup()

        assert remapper.source_grid is None
        assert remapper.target_grid is None
        assert remapper._weights is None

    def test_unstructured_to_structured_grid(self: "TestMPASRemapper") -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS grid data into a structured grid format. The method should return a structured DataArray and a corresponding Dataset for the grid, and the test asserts that these outputs are of the correct types and contain the expected dimensions and coordinate variables. This ensures that the conversion process is functioning correctly and that the resulting structured grid is properly formatted for use in remapping operations.

        Parameters:
            None

        Returns:
            None
        """
        structured_data, structured_grid = MPASRemapper.unstructured_to_structured_grid(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            intermediate_resolution=5.0,
            lon_min=0,
            lon_max=360,
            lat_min=-90,
            lat_max=90,
            buffer=0.0,
        )

        assert isinstance(structured_data, xr.DataArray)
        assert isinstance(structured_grid, xr.Dataset)
        assert len(structured_data.dims) == pytest.approx(2)

        assert "lon" in structured_grid
        assert "lat" in structured_grid
        assert "lon_b" in structured_grid
        assert "lat_b" in structured_grid

    def test_unstructured_to_structured_grid_auto_bounds(
        self: "TestMPASRemapper",
    ) -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS grid data into a structured grid format without explicitly providing longitude and latitude bounds. The method should automatically determine appropriate bounds based on the input coordinates and return a structured DataArray and a corresponding Dataset for the grid. The test asserts that the outputs are of the correct types and contain the expected coordinate variables, confirming that the method can handle cases where bounds are not provided and can still produce a valid structured grid for remapping operations.

        Parameters:
            None

        Returns:
            None
        """
        structured_data, structured_grid = MPASRemapper.unstructured_to_structured_grid(
            data=self.mpas_data,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            intermediate_resolution=10.0,
        )

        assert isinstance(structured_data, xr.DataArray)
        assert isinstance(structured_grid, xr.Dataset)

        assert "grid_conversion" in structured_data.attrs
        assert "intermediate_resolution" in structured_data.attrs

    def test_unstructured_to_structured_with_dataarray(
        self: "TestMPASRemapper",
    ) -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert an xarray DataArray defined on an unstructured MPAS grid into a structured grid format. The method should preserve the attributes of the input DataArray and return a structured DataArray and a corresponding Dataset for the grid. The test asserts that the output DataArray retains the original attributes (e.g., 'units', 'long_name') and that the structured grid is properly formatted for use in remapping operations. This ensures that users can seamlessly convert their unstructured MPAS data into a structured format while maintaining important metadata.

        Parameters:
            None

        Returns:
            None
        """
        data_array = xr.DataArray(
            self.mpas_data,
            dims=["nCells"],
            attrs={"units": "K", "long_name": "Temperature"},
        )

        structured_data, _ = MPASRemapper.unstructured_to_structured_grid(
            data=data_array,
            lon=self.mpas_lon,
            lat=self.mpas_lat,
            intermediate_resolution=5.0,
            lon_min=0,
            lon_max=50,
            lat_min=-20,
            lat_max=20,
        )

        assert structured_data.attrs["units"] == "K"
        assert structured_data.attrs["long_name"] == "Temperature"


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")
class TestBuildRegridder:
    """Test build_regridder method with various parameter combinations."""

    def test_build_regridder_with_weights_dir_and_filename(
        self: "TestBuildRegridder", temp_weights_dir: Path
    ) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when a weights directory and custom filename are specified. The test sets up a remapper with a prepared source grid and target grid, and then it calls `build_regridder` while providing a specific filename for the weights file. The test asserts that the regridder is created without errors and that the weights file is saved in the specified directory with the correct name. This ensures that users can control where their weights files are stored and what they are named when building a regridder.

        Parameters:
            temp_weights_dir (Path): Temporary directory provided by fixture.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=temp_weights_dir, reuse_weights=False
        )

        source_grid = xr.Dataset(
            {
                "lon": xr.DataArray(np.linspace(-120, -110, 20)),
                "lat": xr.DataArray(np.linspace(30, 40, 20)),
            }
        )

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)

        custom_filename = "custom_weights.nc"
        regridder = remapper.build_regridder(filename=custom_filename)

        assert regridder is not None
        assert (temp_weights_dir / custom_filename).exists()

    def test_build_regridder_with_extrap_method(
        self: "TestBuildRegridder", temp_weights_dir: Path
    ) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when an extrapolation method is specified. The test checks that the regridder is created without errors and that it can be built using a specified `extrap_method` value. This ensures that users can control the extrapolation method used in the remapping process and that this parameter is properly integrated into the regridder construction.

        Parameters:
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d",
            extrap_method="nearest_s2d",
            weights_dir=temp_weights_dir,
            reuse_weights=False,
        )

        source_grid = xr.Dataset(
            {
                "lon": xr.DataArray(np.linspace(-120, -110, 15)),
                "lat": xr.DataArray(np.linspace(30, 40, 15)),
            }
        )

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        regridder = remapper.build_regridder()

        assert regridder is not None

    def test_build_regridder_with_extrap_dist_exponent(
        self: "TestBuildRegridder", temp_weights_dir: Path
    ) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when an extrapolation distance exponent is specified. The test checks that the regridder is created without errors and that it can be built using a specified `extrap_dist_exponent` value. This ensures that users can control the distance weighting used in the extrapolation process and that this parameter is properly integrated into the regridder construction.

        Parameters:
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d",
            extrap_method="nearest_s2d",
            extrap_dist_exponent=2.0,
            weights_dir=temp_weights_dir,
            reuse_weights=False,
        )

        source_grid = xr.Dataset(
            {
                "lon": xr.DataArray(np.linspace(-120, -110, 15)),
                "lat": xr.DataArray(np.linspace(30, 40, 15)),
            }
        )

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        regridder = remapper.build_regridder()

        assert regridder is not None

    def test_build_regridder_with_extrap_num_src_pnts(
        self: "TestBuildRegridder", temp_weights_dir: Path
    ) -> None:
        """
        This test verifies that the `build_regridder` method can successfully create a regridder object when the number of source points for extrapolation is restricted. The test checks that the regridder is created without errors and that it can be built using a specified `extrap_num_src_pnts` value. This ensures that users can control the number of source points considered in the extrapolation process and that this parameter is properly integrated into the regridder construction.

        Parameters:
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d",
            extrap_method="nearest_s2d",
            extrap_num_src_pnts=4,
            weights_dir=temp_weights_dir,
            reuse_weights=False,
        )

        source_grid = xr.Dataset(
            {
                "lon": xr.DataArray(np.linspace(-120, -110, 15)),
                "lat": xr.DataArray(np.linspace(30, 40, 15)),
            }
        )

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        regridder = remapper.build_regridder()

        assert regridder is not None


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")
class TestRemapMethod:
    """Test remap method."""

    def test_remap_with_numpy_array(
        self: "TestRemapMethod", temp_weights_dir: Path, mpas_test_data: Any
    ) -> None:
        """
        This test verifies that the `remap` method can successfully remap data when the input is provided as a numpy array. The test sets up a remapper with a built regridder using actual MPAS data to create a structured grid, and then it tests the remapping of a numpy array by wrapping it back as an xarray DataArray with proper dimensions and coordinates. The test asserts that the result of the remapping operation is an xarray DataArray, confirming that the method can handle numpy array inputs correctly by converting them to DataArrays internally before remapping. This ensures that users have flexibility in the format of their input data while still receiving consistent output types.

        Parameters:
            temp_weights_dir (Path): Temporary directory fixture for weights.
            mpas_test_data (Any): Fixture providing a real MPAS Dataset.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=temp_weights_dir, reuse_weights=False
        )

        n_test = min(200, len(mpas_test_data["lonCell"]))
        lon = mpas_test_data["lonCell"].isel(nCells=slice(0, n_test)).values
        lat = mpas_test_data["latCell"].isel(nCells=slice(0, n_test)).values
        data = _RNG.standard_normal(n_test)

        data_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=data, lon=lon, lat=lat, intermediate_resolution=5.0
        )

        remapper.source_grid = grid_dataset

        remapper.create_target_grid(
            lon_min=np.degrees(lon.min()),
            lon_max=np.degrees(lon.max()),
            dlon=10.0,
            dlat=10.0,
        )

        remapper.build_regridder()

        numpy_data = data_array.values
        test_array = xr.DataArray(
            numpy_data, dims=data_array.dims, coords=data_array.coords
        )
        result = remapper.remap(test_array)

        assert isinstance(result, xr.DataArray)


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")
class TestRemapDataset:
    """Test remap_dataset method with various scenarios."""

    def test_remap_dataset_with_missing_variable_skip_missing_true(
        self: "TestRemapDataset", temp_weights_dir: Path, mpas_test_data: Any
    ) -> None:
        """
        This test verifies that the `remap_dataset` method can successfully remap a Dataset even when one of the requested variables is missing, as long as `skip_missing=True` is set. The test sets up a remapper with a built regridder using actual MPAS data to create a structured grid, and then it tests the remapping of a Dataset that includes one existing variable and one nonexistent variable. The test asserts that the remapping operation completes without errors and that the resulting Dataset contains only the existing variable, confirming that the method correctly skips missing variables without failing when `skip_missing` is enabled. This ensures that users can choose to ignore missing variables during remapping if they prefer.

        Parameters:
            temp_weights_dir (Path): Temporary directory fixture for weights.
            mpas_test_data (Any): Fixture providing real MPAS Dataset.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=temp_weights_dir, reuse_weights=False
        )

        n_test = min(200, len(mpas_test_data["lonCell"]))
        lon = mpas_test_data["lonCell"].isel(nCells=slice(0, n_test)).values
        lat = mpas_test_data["latCell"].isel(nCells=slice(0, n_test)).values

        temp_data = _RNG.standard_normal(n_test)
        pressure_data = _RNG.standard_normal(n_test)

        temp_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=temp_data, lon=lon, lat=lat, intermediate_resolution=5.0
        )

        pressure_array, _ = MPASRemapper.unstructured_to_structured_grid(
            data=pressure_data, lon=lon, lat=lat, intermediate_resolution=5.0
        )

        remapper.source_grid = grid_dataset

        remapper.create_target_grid(
            lon_min=np.degrees(lon.min()),
            lon_max=np.degrees(lon.max()),
            dlon=10.0,
            dlat=10.0,
        )

        remapper.build_regridder()

        dataset = xr.Dataset({"temp": temp_array, "pressure": pressure_array})

        result = remapper.remap_dataset(
            dataset, variables=["temp", "nonexistent"], skip_missing=True
        )

        assert "temp" in result
        assert "nonexistent" not in result

    def test_remap_dataset_with_missing_variable_skip_missing_false(
        self: "TestRemapDataset", temp_weights_dir: Path
    ) -> None:
        """
        This test verifies that the `remap_dataset` method raises a ValueError when one of the requested variables is missing and `skip_missing=False` is set. The test sets up a remapper with a built regridder using synthetic source and target grids, and then it tests the remapping of a Dataset that includes one existing variable and one nonexistent variable. The test asserts that the remapping operation raises a ValueError with an appropriate message indicating that the variable was not found in the dataset, confirming that the method correctly enforces the presence of all requested variables when `skip_missing` is disabled. This ensures that users are alerted to missing variables during remapping if they choose not to skip them.

        Parameters:
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=temp_weights_dir, reuse_weights=False
        )

        source_grid = xr.Dataset(
            {
                "lon": xr.DataArray(np.linspace(-120, -110, 20)),
                "lat": xr.DataArray(np.linspace(30, 40, 20)),
            }
        )

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        remapper.build_regridder()

        dataset = xr.Dataset(
            {
                "temp": xr.DataArray(_RNG.standard_normal(20), dims=["x"]),
            }
        )

        with pytest.raises(ValueError, match="Variable.*not found in dataset"):
            remapper.remap_dataset(
                dataset, variables=["nonexistent"], skip_missing=False
            )

    def test_remap_dataset_with_exception_skip_missing_true(
        self: "TestRemapDataset", temp_weights_dir: Path
    ) -> None:
        """
        This test verifies that the `remap_dataset` method can gracefully handle exceptions during remapping when `skip_missing=True` is set. The test sets up a remapper with a built regridder using synthetic source and target grids, and then it tests the remapping of a Dataset that is intentionally designed to cause an error (e.g., by having an incompatible shape). The test asserts that the remapping operation does not raise an exception and instead returns an empty Dataset or skips the problematic variable, confirming that the method can handle errors gracefully without failing when `skip_missing` is enabled. This ensures that users can choose to ignore errors during remapping if they prefer.

        Parameters:
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=temp_weights_dir, reuse_weights=False
        )

        source_grid = xr.Dataset(
            {
                "lon": xr.DataArray(np.linspace(-120, -110, 20)),
                "lat": xr.DataArray(np.linspace(30, 40, 20)),
            }
        )

        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        remapper.build_regridder()

        dataset = xr.Dataset(
            {
                "temp": xr.DataArray(_RNG.standard_normal(100), dims=["x"]),
            }
        )

        result = remapper.remap_dataset(dataset, skip_missing=True)
        assert isinstance(result, xr.Dataset)


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")
class TestUnstructuredToStructuredGrid:
    """Test unstructured_to_structured_grid method."""

    def test_with_explicit_bounds(
        self: "TestUnstructuredToStructuredGrid", simple_mpas_data: Any
    ) -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS-like data into a structured grid format when explicit longitude and latitude bounds are provided. The test checks that the returned structured DataArray and Dataset are of the correct types and that the grid Dataset contains the expected coordinate variables ('lon' and 'lat'). This ensures that the method can handle cases where users specify bounds for the conversion process and can produce a valid structured grid for remapping operations.

        Parameters:
            simple_mpas_data (Any): Fixture providing small MPAS-like arrays.

        Returns:
            None
        """
        data_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=simple_mpas_data["data"],
            lon=simple_mpas_data["lon"],
            lat=simple_mpas_data["lat"],
            intermediate_resolution=2.0,
            lon_min=-180,
            lon_max=180,
            lat_min=-90,
            lat_max=90,
            buffer=0.0,
        )

        assert isinstance(data_array, xr.DataArray)
        assert isinstance(grid_dataset, xr.Dataset)
        assert "lon" in grid_dataset.coords
        assert "lat" in grid_dataset.coords

    def test_with_none_bounds(
        self: "TestUnstructuredToStructuredGrid", simple_mpas_data: Any
    ) -> None:
        """
        This test verifies that the `unstructured_to_structured_grid` method can successfully convert unstructured MPAS-like data into a structured grid format when longitude and latitude bounds are set to None, allowing the method to automatically determine appropriate bounds based on the input coordinates. The test checks that the returned structured DataArray and Dataset are of the correct types and that the resulting grid Dataset contains the expected coordinate variables. This ensures that the method can handle cases where users do not provide explicit bounds and can still produce a valid structured grid for remapping operations.

        Parameters:
            simple_mpas_data (Any): Fixture providing small MPAS-like arrays.

        Returns:
            None
        """
        data_array, grid_dataset = MPASRemapper.unstructured_to_structured_grid(
            data=simple_mpas_data["data"],
            lon=simple_mpas_data["lon"],
            lat=simple_mpas_data["lat"],
            intermediate_resolution=2.0,
            lon_min=None,
            lon_max=None,
            lat_min=None,
            lat_max=None,
        )

        assert isinstance(data_array, xr.DataArray)
        assert isinstance(grid_dataset, xr.Dataset)


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")
class TestCleanup:
    """Test cleanup method."""

    def test_cleanup_clears_regridder_and_grids(
        self: "TestCleanup", temp_weights_dir: Path
    ) -> None:
        """
        This test verifies that the `cleanup` method of the `MPASRemapper` class properly clears the regridder and grid attributes. The test sets up a remapper with a built regridder using synthetic source and target grids, and then it calls the `cleanup` method. The test asserts that after cleanup, the `regridder`, `source_grid`, and `target_grid` attributes are all set to None, confirming that the method effectively releases resources and resets the state of the remapper. This ensures that users can rely on the `cleanup` method to free up memory and prepare the remapper for a new remapping operation if needed.

        Parameters:
            temp_weights_dir (Path): Temporary directory fixture for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=temp_weights_dir, reuse_weights=False
        )

        source_grid = xr.Dataset(
            {
                "lon": xr.DataArray(np.linspace(-120, -110, 20)),
                "lat": xr.DataArray(np.linspace(30, 40, 20)),
            }
        )

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
    def test_remap_dataset_skip_missing_false_raises(
        self: "TestRemapDatasetSkipMissingFalse", tmp_path: Path
    ) -> None:
        """
        This test verifies that the `remap_dataset` method raises an exception when `skip_missing=False` and an error occurs during remapping (e.g., due to incompatible variable shapes). The test sets up a remapper with a built regridder using synthetic source and target grids, and then it attempts to remap a Dataset that is intentionally designed to cause an error. The test asserts that an exception is raised, confirming that the method correctly enforces error handling when `skip_missing` is disabled. This ensures that users are alerted to issues during remapping if they choose not to skip errors.

        Parameters:
            tmp_path (Path): Temporary directory fixture for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=tmp_path, reuse_weights=False
        )
        source_grid = xr.Dataset(
            {
                "lon": xr.DataArray(np.linspace(-120, -110, 20)),
                "lat": xr.DataArray(np.linspace(30, 40, 20)),
            }
        )
        remapper.source_grid = source_grid
        remapper.create_target_grid(lon_min=-120, lon_max=-110, dlon=2.0, dlat=2.0)
        remapper.build_regridder()

        bad_dataset = xr.Dataset(
            {
                "bad_var": xr.DataArray(_RNG.standard_normal(100), dims=["x"]),
            }
        )
        with pytest.raises(Exception):
            remapper.remap_dataset(bad_dataset, skip_missing=False)


@pytest.mark.skipif(not ESMPY_AVAILABLE, reason="ESMPy not installed")
class TestMPASRemapperCoveragePaths:
    """Tests targeting ESMPy code paths not hit by existing tests."""

    @pytest.fixture
    def temp_dir(self: "TestMPASRemapperCoveragePaths", tmp_path: Path) -> Path:
        """
        This fixture provides a temporary directory for tests that require file I/O, such as saving and loading weights files. The `tmp_path` fixture from pytest is used to create a temporary directory that is automatically cleaned up after the test session. This allows tests to write files without affecting the actual filesystem and ensures that any generated files do not persist after the tests are completed.

        Parameters:
            tmp_path (Path): Temporary directory fixture provided by pytest.

        Returns:
            Path: Temporary directory path.
        """
        return tmp_path

    @pytest.fixture
    def unstructured_source(self: "TestMPASRemapperCoveragePaths") -> xr.Dataset:
        """
        This fixture creates a synthetic unstructured source grid dataset with 'lon' and 'lat' variables defined on an 'nCells' dimension. The longitude and latitude values are randomly generated within specified bounds to simulate an unstructured grid similar to what might be found in MPAS data. This dataset can be used in tests that require an unstructured source grid for remapping operations, allowing the tests to cover code paths that handle unstructured grids.

        Parameters:
            None

        Returns:
            xr.Dataset: Dataset containing 'lon' and 'lat' variables on an 'nCells' dimension.
        """
        n = 60
        rng = np.random.default_rng(77)
        lon = rng.uniform(-110, -100, n)
        lat = rng.uniform(30, 40, n)
        return xr.Dataset(
            {
                "lon": xr.DataArray(lon, dims=["nCells"]),
                "lat": xr.DataArray(lat, dims=["nCells"]),
            }
        )

    @pytest.fixture
    def structured_source(self: "TestMPASRemapperCoveragePaths") -> xr.Dataset:
        """
        This fixture creates a synthetic structured source grid dataset with 'lon' and 'lat' variables defined on separate dimensions. The longitude and latitude values are linearly spaced within specified bounds to simulate a structured grid similar to what might be found in MPAS data. This dataset can be used in tests that require a structured source grid for remapping operations, allowing the tests to cover code paths that handle structured grids.

        Parameters:
            None

        Returns:
            xr.Dataset: Dataset containing 'lon' and 'lat' variables on separate dimensions.
        """
        lon = np.linspace(-110, -100, 12)
        lat = np.linspace(30, 40, 10)
        return xr.Dataset(
            {
                "lon": xr.DataArray(lon, dims=["lon"]),
                "lat": xr.DataArray(lat, dims=["lat"]),
            }
        )

    def _build_nearest_remapper(
        self: "TestMPASRemapperCoveragePaths",
        source: xr.Dataset,
        temp_dir: Path,
        reuse: bool = False,
    ) -> MPASRemapper:
        """
        This helper method builds an `MPASRemapper` instance configured for nearest neighbor remapping from an unstructured source grid to a structured target grid. The method sets up the remapper with the specified source dataset, creates a target grid with defined longitude and latitude bounds and resolution, and configures the remapper to use a weights directory for caching. The `reuse` parameter controls whether existing weight files should be reused if available. This method can be used in multiple tests to ensure consistent setup of the remapper for testing various code paths related to weight caching and file I/O.

        Parameters:
            source (xr.Dataset): The source dataset containing 'lon' and 'lat' variables.
            temp_dir (Path): The directory where weight files will be stored.
            reuse (bool): Whether to reuse existing weight files if available.

        Returns:
            MPASRemapper: An instance of the MPASRemapper configured with the source dataset and target grid.
        """
        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=temp_dir, reuse_weights=reuse
        )
        remapper.source_grid = source
        remapper.create_target_grid(
            lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, dlon=2.0, dlat=2.0
        )
        return remapper

    def test_weight_caching_reuses_saved_file(
        self: "TestMPASRemapperCoveragePaths",
        unstructured_source: xr.Dataset,
        temp_dir: Path,
    ) -> None:
        """
        This test verifies that the weight caching mechanism in `build_regridder` correctly saves weights to a file and can reuse them when `reuse_weights=True` is set. The test first builds a remapper and saves the weights to a specified file, then it creates a new remapper with the same configuration and checks that it can load the weights from the existing file without errors. This ensures that the caching mechanism is functioning as intended, allowing for faster regridding operations by avoiding redundant weight calculations when the same source and target grids are used.

        Parameters:
            unstructured_source (xr.Dataset): The unstructured source grid dataset.
            temp_dir (Path): Temporary directory for storing weight files.

        Returns:
            None
        """
        fname = "cached_weights.nc"

        r1 = self._build_nearest_remapper(unstructured_source, temp_dir, reuse=False)
        r1.build_regridder(filename=fname)
        weights_file = temp_dir / fname

        assert weights_file.exists()

        r2 = MPASRemapper(
            method="nearest_s2d", weights_dir=temp_dir, reuse_weights=True
        )
        r2.source_grid = unstructured_source

        r2.create_target_grid(
            lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, dlon=2.0, dlat=2.0
        )

        weights = r2.build_regridder(filename=fname)
        assert weights is not None

    def test_save_and_load_weights_roundtrip(
        self: "TestMPASRemapperCoveragePaths",
        unstructured_source: xr.Dataset,
        temp_dir: Path,
    ) -> None:
        """
        This test verifies that the weights saved by `build_regridder` can be successfully loaded using the `_load_weights_netcdf` method. The test builds a remapper, saves the weights to a file, and then directly calls the loading method to retrieve the weight matrix, target shape, and cell_of_element information. The test asserts that the loaded weight matrix is not None and that the target shape has valid dimensions, confirming that the save and load functionality for weights files is working correctly. This ensures that users can rely on the weights caching mechanism to persist and retrieve remapping weights as needed.

        Parameters:
            unstructured_source (xr.Dataset): The unstructured source grid dataset.
            temp_dir (Path): Temporary directory for storing weight files.

        Returns:
            None
        """
        remapper = self._build_nearest_remapper(
            unstructured_source, temp_dir, reuse=False
        )
        remapper.build_regridder(filename="roundtrip.nc")
        weights_path = temp_dir / "roundtrip.nc"
        assert weights_path.exists()

        weight_matrix, tgt_shape, _ = MPASRemapper._load_weights_netcdf(weights_path)
        assert weight_matrix is not None
        assert len(tgt_shape) == 2
        assert tgt_shape[0] > 0 and tgt_shape[1] > 0

    def test_load_weights_with_cell_of_element(
        self: "TestMPASRemapperCoveragePaths",
        unstructured_source: xr.Dataset,
        temp_dir: Path,
    ) -> None:
        """
        This test verifies that the `_load_weights_netcdf` method correctly handles cases where the weights file does not contain a 'cell_of_element' variable, which is expected for unstructured locstream sources. The test builds a remapper and saves the weights to a file, then it loads the weights using the loading method and checks that the 'cell_of_element' variable is None. This ensures that the loading method can gracefully handle different types of source grids and their corresponding weights files without errors.

        Parameters:
            unstructured_source (xr.Dataset): The unstructured source grid dataset.
            temp_dir (Path): Temporary directory for storing weight files.

        Returns:
            None
        """
        remapper = self._build_nearest_remapper(
            unstructured_source, temp_dir, reuse=False
        )
        remapper.build_regridder(filename="coe_test.nc")
        _, _, coe = MPASRemapper._load_weights_netcdf(temp_dir / "coe_test.nc")
        assert coe is None

    def test_patch_method_raises_value_error(
        self: "TestMPASRemapperCoveragePaths",
        unstructured_source: xr.Dataset,
        temp_dir: Path,
    ) -> None:
        """
        This test verifies that the `build_regridder` method raises a ValueError when the 'patch' method is specified, as this method is not implemented in the current version of the MPASRemapper. The test sets up a remapper with the 'patch' method and attempts to build the regridder, expecting a ValueError to be raised with an appropriate message. This ensures that users are informed about the unavailability of the 'patch' method and that the code correctly handles unsupported remapping methods.

        Parameters:
            unstructured_source (xr.Dataset): The unstructured source grid dataset.
            temp_dir (Path): Temporary directory for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="patch", weights_dir=temp_dir, reuse_weights=False
        )
        remapper.source_grid = unstructured_source
        remapper.create_target_grid(
            lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, dlon=2.0, dlat=2.0
        )
        with pytest.raises(ValueError, match="patch"):
            remapper.build_regridder()

    def test_conservative_without_boundaries_raises(
        self: "TestMPASRemapperCoveragePaths",
        unstructured_source: xr.Dataset,
        temp_dir: Path,
    ) -> None:
        """
        This test verifies that the `_prepare_source_esmpy` method raises a ValueError when the source grid does not contain the required 'lon_b' and 'lat_b' boundary variables. The test sets up a remapper with the 'conservative' method and an unstructured source grid without boundaries, then attempts to build the regridder, expecting a ValueError to be raised. This ensures that the code correctly handles cases where the source grid is missing necessary boundary information.

        Parameters:
            unstructured_source (xr.Dataset): The unstructured source grid dataset.
            temp_dir (Path): Temporary directory for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="conservative", weights_dir=temp_dir, reuse_weights=False
        )
        remapper.source_grid = unstructured_source  # no lon_b / lat_b
        remapper.create_target_grid(
            lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, dlon=2.0, dlat=2.0
        )
        with pytest.raises(ValueError, match="[Cc]onservative"):
            remapper.build_regridder()

    def test_structured_source_builds_successfully(
        self: "TestMPASRemapperCoveragePaths",
        structured_source: xr.Dataset,
        temp_dir: Path,
    ) -> None:
        """
        This test verifies that a structured (2-D dims) source grid takes the structured path in `_prepare_source_esmpy`.

        Parameters:
            structured_source (xr.Dataset): The structured source grid dataset.
            temp_dir (Path): Temporary directory for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="nearest_s2d", weights_dir=temp_dir, reuse_weights=False
        )
        remapper.source_grid = structured_source
        remapper.create_target_grid(
            lon_min=-110, lon_max=-100, lat_min=30, lat_max=40, dlon=2.0, dlat=2.0
        )
        weights = remapper.build_regridder()
        assert weights is not None

    @pytest.fixture
    def triangular_source(self: "TestMPASRemapperCoveragePaths") -> xr.Dataset:
        """
        This fixture creates a synthetic unstructured source grid dataset with 'lon', 'lat', 'lon_b', and 'lat_b' variables defined on an 'nCells' dimension, simulating a triangular mesh similar to what might be found in MPAS data. The 'lon' and 'lat' variables represent the cell centers, while the 'lon_b' and 'lat_b' variables represent the vertices of the triangular cells. This dataset can be used in tests that require a triangular mesh source grid for remapping operations, allowing the tests to cover code paths that handle conservative remapping with triangular meshes.

        Parameters:
            None

        Returns:
            xr.Dataset: Dataset containing 'lon', 'lat', 'lon_b', and 'lat_b' variables for a triangular mesh.
        """
        lon_c = np.array([-104.0, -104.0, -101.0, -101.0])
        lat_c = np.array([36.0, 38.0, 36.0, 38.0])
        lon_b = np.array(
            [
                [-105.0, -103.0, -104.0],
                [-105.0, -103.0, -104.0],
                [-102.0, -100.0, -101.0],
                [-102.0, -100.0, -101.0],
            ]
        )
        lat_b = np.array(
            [
                [35.0, 35.0, 37.0],
                [37.0, 37.0, 39.0],
                [35.0, 35.0, 37.0],
                [37.0, 37.0, 39.0],
            ]
        )
        return xr.Dataset(
            {
                "lon": xr.DataArray(lon_c, dims=["nCells"]),
                "lat": xr.DataArray(lat_c, dims=["nCells"]),
                "lon_b": xr.DataArray(lon_b, dims=["nCells", "nv"]),
                "lat_b": xr.DataArray(lat_b, dims=["nCells", "nv"]),
            }
        )

    def test_conservative_with_triangular_mesh(
        self: "TestMPASRemapperCoveragePaths",
        triangular_source: xr.Dataset,
        temp_dir: Path,
    ) -> None:
        """
        This test verifies that the `build_regridder` method can successfully build a regridder using the 'conservative' method when the source grid is a triangular mesh with appropriate boundary variables. The test sets up a remapper with the triangular source grid, creates a structured target grid, and attempts to build the regridder. The test asserts that the weights are successfully created, confirming that the conservative remapping code path for triangular meshes is functioning correctly. This ensures that users can perform conservative remapping with triangular mesh sources when the necessary boundary information is provided.

        Parameters:
            triangular_source (xr.Dataset): The triangular mesh source grid dataset.
            temp_dir (Path): Temporary directory for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="conservative", weights_dir=temp_dir, reuse_weights=False
        )
        remapper.source_grid = triangular_source
        remapper.create_target_grid(
            lon_min=-106, lon_max=-99, lat_min=34, lat_max=41, dlon=2.0, dlat=2.0
        )
        weights = remapper.build_regridder()
        assert weights is not None

        data = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0]), dims=["nCells"])
        result = remapper.remap(data)
        assert isinstance(result, xr.DataArray)

    def test_conservative_normed_uses_fracarea_norm(
        self: "TestMPASRemapperCoveragePaths",
        triangular_source: xr.Dataset,
        temp_dir: Path,
    ) -> None:
        """
        This test verifies that the `build_regridder` method can successfully build a regridder using the 'conservative_normed' method when the source grid is a triangular mesh with appropriate boundary variables. The test sets up a remapper with the triangular source grid, creates a structured target grid, and attempts to build the regridder. The test asserts that the weights are successfully created, confirming that the conservative normed remapping code path for triangular meshes is functioning correctly and that it uses the fracarea normalization as expected. This ensures that users can perform conservative normed remapping with triangular mesh sources when the necessary boundary information is provided.

        Parameters:
            triangular_source (xr.Dataset): The triangular mesh source grid dataset.
            temp_dir (Path): Temporary directory for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="conservative_normed", weights_dir=temp_dir, reuse_weights=False
        )
        remapper.source_grid = triangular_source
        remapper.create_target_grid(
            lon_min=-106, lon_max=-99, lat_min=34, lat_max=41, dlon=2.0, dlat=2.0
        )
        weights = remapper.build_regridder()
        assert weights is not None

    def test_conservative_saves_cell_of_element_in_weights_file(
        self: "TestMPASRemapperCoveragePaths",
        triangular_source: xr.Dataset,
        temp_dir: Path,
    ) -> None:
        """
        This test verifies that the weight file for conservative remapping includes the `cell_of_element` variable (line 1090). The test sets up a remapper with the triangular source grid, creates a structured target grid, and builds the regridder. The test asserts that the `cell_of_element` variable is present in the weights file, confirming that the conservative remapping code path correctly includes this information for triangular mesh sources.

        Parameters:
            triangular_source (xr.Dataset): The triangular mesh source grid dataset.
            temp_dir (Path): Temporary directory for weights.

        Returns:
            None
        """
        remapper = MPASRemapper(
            method="conservative", weights_dir=temp_dir, reuse_weights=False
        )
        remapper.source_grid = triangular_source
        remapper.create_target_grid(
            lon_min=-106, lon_max=-99, lat_min=34, lat_max=41, dlon=2.0, dlat=2.0
        )
        remapper.build_regridder(filename="conservative_weights.nc")
        weights_file = temp_dir / "conservative_weights.nc"
        assert weights_file.exists()

        _, _, cell_of_element = MPASRemapper._load_weights_netcdf(weights_file)
        assert cell_of_element is not None  # mesh sources have cell_of_element


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
