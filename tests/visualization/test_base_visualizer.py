#!/usr/bin/env python3
"""
MPASdiag Test Suite: Base Processor and Visualizer Tests

This module contains unit tests for the MPASBaseProcessor and MPASVisualizer classes, covering initialization, file discovery, spatial coordinate handling, dataset operations, and helper methods. The tests use pytest fixtures to set up temporary resources and mock data, ensuring that the processor and visualizer can handle various scenarios including real MPAS data when available. The tests also verify that verbose output is generated correctly and that exceptions are raised with informative messages when expected.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import sys
import glob
import pytest
import shutil
import tempfile
import warnings
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Generator
from cartopy.mpl.geoaxes import GeoAxes
from unittest.mock import Mock, patch

from mpasdiag.processing.base import MPASBaseProcessor
from mpasdiag.visualization.base_visualizer import MPASVisualizer
from tests.test_data_helpers import load_mpas_coords_from_processor

warnings.filterwarnings('ignore')

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')
MPASOUT_DIR = os.path.join(TEST_DATA_DIR, 'u240k', 'mpasout')


class TestProcessorInitializationAndSetup:
    """ Tests for MPASBaseProcessor initialization and basic setup. Consolidates tests for processor instantiation, initialization parameters, and basic configuration validation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestProcessorInitializationAndSetup", mock_mpas_mesh) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing the initialization of MPASBaseProcessor. It creates a temporary directory and grid file using the provided mock_mpas_mesh fixture, which contains real MPAS mesh data when available. The fixture yields control to the test methods and performs cleanup after the tests complete, ensuring that the processor can be initialized with a valid grid file and that its attributes are set correctly.

        Parameters:
            mock_mpas_mesh: Fixture providing real or synthetic MPAS mesh data.

        Returns:
            Generator[None, None, None]: Yields control to the test and performs cleanup after the test completes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "test_grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)
        
        yield
        
        processor = MPASBaseProcessor(self.grid_file, verbose=False)
        
        assert processor.grid_file == self.grid_file
        assert not processor.verbose
        assert processor.dataset is None
        assert processor.data_type is None
    
    def test_init_verbose_mode(self: "TestProcessorInitializationAndSetup") -> None:
        """
        This test verifies that when the `MPASBaseProcessor` is initialized with `verbose=True`, the `verbose` attribute of the processor instance is set to True. This ensures that the processor will emit verbose output during its operations, which can be helpful for debugging and understanding the processing steps.

        Parameters:
            self (Any): Test case instance providing `grid_file` fixture.

        Returns:
            None: Assertion validates `processor.verbose` is True.
        """
        processor = MPASBaseProcessor(self.grid_file, verbose=True)
        
        assert processor.verbose
    
    def test_init_nonexistent_grid_file(self: "TestProcessorInitializationAndSetup") -> None:
        """
        This test ensures that initializing `MPASBaseProcessor` with a non-existent grid file path raises a `FileNotFoundError`. The test asserts that the exception message contains "Grid file not found" to confirm that the error is informative and helps users identify the issue with the provided grid file path.

        Parameters:
            self (Any): Test case instance context.

        Returns:
            None: Assertion validates that FileNotFoundError is raised.
        """
        with pytest.raises(FileNotFoundError) as ctx:
            MPASBaseProcessor("/nonexistent/grid.nc", verbose=False)
        
        assert "Grid file not found" in str(ctx.value)


class TestFileDiscoveryAndValidation:
    """ Tests for file discovery, pattern matching, and validation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestFileDiscoveryAndValidation", mock_mpas_mesh) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing file discovery and validation methods of the `MPASBaseProcessor`. It creates a temporary directory and populates it with test files that match the expected glob pattern. The fixture uses the provided `mock_mpas_mesh` to create real MPAS mesh files for testing. After yielding control to the test methods, it performs cleanup by removing the temporary directory and its contents. The fixture also includes a test to ensure that the `validate_files` method correctly handles an empty list of files without raising an error.

        Parameters:
            mock_mpas_mesh: Fixture providing real or synthetic MPAS mesh data.

        Returns:
            Generator[None, None, None]: Yields control to the test and performs cleanup after the test completes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)        
        self.processor = MPASBaseProcessor(self.grid_file, verbose=False)
        
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir)
        
        for i in range(5):
            file_path = os.path.join(self.data_dir, f"diag.2024-01-0{i+1}_00.00.00.nc")
            mock_mpas_mesh.to_netcdf(file_path)
    
        yield
        
        result = self.processor.validate_files([])
        assert result == []
        
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_find_files_by_pattern_verbose(self: "TestFileDiscoveryAndValidation") -> None:
        """
        This test verifies that the `_find_files_by_pattern` method correctly identifies files matching the specified glob pattern and prints verbose output when `processor.verbose` is True. The test asserts that the expected number of files is found, that they all have the correct file extension, and that the output messages indicate the discovery of diagnostic files. The test also checks that the returned file list is sorted, which is important for consistent processing order.

        Parameters:
            self (Any): Test case instance providing `processor` and test files.

        Returns:
            None: Assertions validate stdout messages and returned file list.
        """
        self.processor.verbose = True
        
        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            files = self.processor._find_files_by_pattern(
                self.data_dir, "diag*.nc", "diagnostic files"
            )
            output = captured_output.getvalue()
            
            assert "Found 5 diagnostic files" in output
            assert len(files) == pytest.approx(5)
            assert all(f.endswith('.nc') for f in files)
            assert files == sorted(files)
        finally:
            sys.stdout = sys.__stdout__
    
    def test_find_files_no_matches(self: "TestFileDiscoveryAndValidation") -> None:
        """
        This test ensures that the `_find_files_by_pattern` method raises a `FileNotFoundError` when no files match the specified glob pattern. The test asserts that the exception message contains "No test files found" to confirm that the error is informative and helps users understand that the expected files were not discovered in the provided directory.

        Parameters:
            self (Any): Test case instance providing test directory.

        Returns:
            None: Assertion verifies FileNotFoundError message content.
        """
        with pytest.raises(FileNotFoundError) as ctx:
            self.processor._find_files_by_pattern(
                self.data_dir, "nonexistent*.nc", "test files"
            )
        
        assert "No test files found" in str(ctx.value)
    
    def test_find_files_insufficient_files(self: "TestFileDiscoveryAndValidation") -> None:
        """
        This test verifies that the `_find_files_by_pattern` method raises a `ValueError` when the number of discovered files is insufficient for processing. The test simulates this scenario by removing files from the temporary test directory and asserts that the raised exception includes guidance about the minimum required files.

        Parameters:
            self (Any): Test case instance with temporary test files.

        Returns:
            None: Assertion verifies the ValueError message includes guidance.
        """
        files = sorted(glob.glob(os.path.join(self.data_dir, "diag*.nc")))

        for f in files[1:]:
            os.remove(f)
        
        with pytest.raises(ValueError) as ctx:
            self.processor._find_files_by_pattern(
                self.data_dir, "diag*.nc", "diagnostic files"
            )
        
        assert "Insufficient files" in str(ctx.value)
        assert "need at least 2" in str(ctx.value)
    
    def test_validate_files_all_valid(self: "TestFileDiscoveryAndValidation") -> None:
        """
        This test confirms that the `validate_files` method correctly identifies all provided files as valid when they exist and are readable. The test uses the temporary test files created in the fixture and asserts that the returned list of valid files matches the input list, indicating that all files passed validation.

        Parameters:
            self (Any): Test case instance with created diagnostic files.

        Returns:
            None: Assertion verifies returned list length equals input.
        """
        files = glob.glob(os.path.join(self.data_dir, "diag*.nc"))
        
        valid_files = self.processor.validate_files(files)
        
        assert len(valid_files) == len(files)
    
    def test_validate_files_nonexistent(self: "TestFileDiscoveryAndValidation") -> None:
        """
        This test ensures that the `validate_files` method raises a `FileNotFoundError` when one or more provided file paths do not exist. The test asserts that the exception message contains "File not found" to confirm that the error is informative and helps users identify which files are missing.

        Parameters:
            self (Any): Test case instance context.

        Returns:
            None: Assertion verifies FileNotFoundError message includes 'File not found'.
        """
        files = ["/nonexistent/file.nc"]
        
        with pytest.raises(FileNotFoundError) as ctx:
            self.processor.validate_files(files)
        
        assert "File not found" in str(ctx.value)
    
    def test_validate_files_not_readable(self: "TestFileDiscoveryAndValidation") -> None:
        """
        This test verifies that the `validate_files` method treats unreadable files as missing and raises a `FileNotFoundError`. The test creates a temporary file and removes read permissions to simulate an unreadable filesystem entry.

        Parameters:
            self (Any): Test case instance with temporary filesystem entries.

        Returns:
            None: Assertion verifies the raised error indicates unreadable file.
        """
        test_file = os.path.join(self.temp_dir, "readonly.nc")
        
        if hasattr(self, 'processor'):
            ds = xr.Dataset({'dummy': xr.DataArray([1])})
            ds.to_netcdf(test_file)
        
            os.chmod(test_file, 0o000)
        
            try:
                with pytest.raises(FileNotFoundError) as ctx:
                    self.processor.validate_files([test_file])
            
                assert "not readable" in str(ctx.value)
            finally:
                os.chmod(test_file, 0o600)


class TestSpatialCoordinatesAndFiltering:
    """ Tests for spatial coordinate extraction, normalization, and geographic filtering. """

    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestSpatialCoordinatesAndFiltering", mock_mpas_mesh) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing spatial coordinate handling and geographic filtering methods of the `MPASBaseProcessor`. It creates a temporary directory and grid file using the provided `mock_mpas_mesh` fixture, which contains real MPAS mesh data when available. The fixture initializes an instance of `MPASBaseProcessor` with the created grid file and yields control to the test methods. After the tests complete, it performs cleanup by removing the temporary directory and its contents. This setup allows the tests to exercise real spatial coordinate data and validate the processor's handling of geographic extents and coordinate normalization.

        Parameters:
            mock_mpas_mesh: Fixture providing real or synthetic MPAS mesh data.

        Returns:
            Generator[None, None, None]: Yields control to the test and performs teardown after the test completes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)        
        self.processor = MPASBaseProcessor(self.grid_file, verbose=False)
        
        yield
        
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_geographic_extent_invalid_lon(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test ensures that the `validate_geographic_extent` method correctly identifies longitude values that are outside the valid range of [-180, 180] degrees. The test provides examples of geographic extents with invalid longitude bounds and asserts that the method returns False, indicating that the provided extent is not valid.

        Parameters:
            None

        Returns:
            None
        """
        result = self.processor.validate_geographic_extent((-200, -80, 30, 50))
        assert not result

        result = self.processor.validate_geographic_extent((-120, 200, 30, 50))
        assert not result
    
    def test_validate_geographic_extent_invalid_lat(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test verifies that the `validate_geographic_extent` method correctly identifies latitude values that are outside the valid range of [-90, 90] degrees. The test provides examples of geographic extents with invalid latitude bounds and asserts that the method returns False, indicating that the provided extent is not valid.

        Parameters:
            None

        Returns:
            None
        """
        result = self.processor.validate_geographic_extent((-120, -80, -100, 50))
        assert not result

        result = self.processor.validate_geographic_extent((-120, -80, 30, 100))
        assert not result
    
    def test_validate_geographic_extent_reversed_bounds(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test confirms that the `validate_geographic_extent` method detects when the minimum longitude is greater than the maximum longitude or when the minimum latitude is greater than the maximum latitude. The test provides examples of geographic extents with reversed bounds and asserts that the method returns False, indicating that the provided extent is not valid.

        Parameters:
            None

        Returns:
            None
        """
        result = self.processor.validate_geographic_extent((-80, -120, 30, 50))
        assert not result

        result = self.processor.validate_geographic_extent((-120, -80, 50, 30))
        assert not result

    def test_normalize_longitude_array(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test verifies that the `normalize_longitude` method correctly normalizes an array of longitude values to the range [-180, 180] degrees. The test provides an array of longitude values that includes values outside the valid range and asserts that the returned array is of type `numpy.ndarray` and contains the expected normalized values. The test checks that values greater than 180 are wrapped around to the negative range and that values at the boundaries are handled correctly.

        Parameters:
            self (Any): Test case instance with `processor` fixture.

        Returns:
            None: Assertions validate array type and normalized values.
        """
        lons = np.array([0, 90, 180, 270, 360])
        result = self.processor.normalize_longitude(lons)
        
        assert isinstance(result, np.ndarray)
        expected = np.array([0, 90, -180, -90, 0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_normalize_longitude_negative(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test verifies that the `normalize_longitude` method correctly handles scalar negative longitude values and returns a float in the expected normalized range. This ensures scalar inputs are supported in addition to array inputs. The test provides a negative longitude value and asserts that the returned value is approximately equal to the expected normalized value, confirming that the method correctly wraps negative longitudes into the valid range.

        Parameters:
            self (Any): Test case instance with `processor` fixture.

        Returns:
            None: Assertion validates the returned scalar equals expected.
        """
        result = self.processor.normalize_longitude(-90.0)
        assert result == pytest.approx(-90.0)
    
    def test_extract_spatial_coordinates_degrees(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test verifies that the `extract_spatial_coordinates` method correctly returns longitude and latitude arrays when the dataset stores coordinates in degrees. The test provides a dataset with `lonCell` and `latCell` and asserts that the output arrays have the expected lengths and valid ranges.

        Parameters:
            self (Any): Test case instance with a processor fixture.

        Returns:
            None: Assertions verify lengths and value ranges of returned arrays.
        """
        self.processor.dataset = xr.Dataset({
            'lonCell': xr.DataArray(np.array([0, 90, 180, 270]), dims=['nCells']),
            'latCell': xr.DataArray(np.array([-45, 0, 45, 90]), dims=['nCells'])
        })
        
        lon, lat = self.processor.extract_spatial_coordinates()
        
        assert len(lon) == pytest.approx(4)
        assert len(lat) == pytest.approx(4)
        assert np.all(lon >= -180.0)
        assert np.all(lon <= 180.0)
    
    def test_extract_spatial_coordinates_radians(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test verifies that the `extract_spatial_coordinates` method correctly converts radian-based coordinate variables into degrees. The test provides `lonCell` and `latCell` values in radians and asserts that the returned coordinates are converted to degree values with expected maxima.

        Parameters:
            self (Any): Test case instance with processor fixture.

        Returns:
            None: Assertions validate conversion to degrees and expected maxima.
        """
        self.processor.dataset = xr.Dataset({
            'lonCell': xr.DataArray(np.array([0, np.pi/2, np.pi, -np.pi/2]), dims=['nCells']),
            'latCell': xr.DataArray(np.array([-np.pi/4, 0, np.pi/4, np.pi/2]), dims=['nCells'])
        })
        
        lon, lat = self.processor.extract_spatial_coordinates()
        
        assert np.max(np.abs(lat)) > np.pi
        assert np.max(lat) == pytest.approx(90.0, abs=1e-5)
    
    def test_extract_spatial_coordinates_no_dataset(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test verifies that the `extract_spatial_coordinates` method raises a `ValueError` when no dataset is loaded on the processor. The test ensures that callers receive a clear error indicating the missing dataset.

        Parameters:
            self (Any): Test case instance with an empty processor.

        Returns:
            None: Assertion verifies ValueError contains 'Dataset not loaded'.
        """
        with pytest.raises(ValueError) as ctx:
            self.processor.extract_spatial_coordinates()
        
        assert "Dataset not loaded" in str(ctx.value)
    
    def test_extract_spatial_coordinates_missing_coords(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test verifies that the `extract_spatial_coordinates` method raises a `ValueError` when expected coordinate variables are absent from the dataset. The test provides a dataset with unrelated variables and asserts the function reports the missing coordinates.

        Parameters:
            self (Any): Test case instance with a processor fixture.

        Returns:
            None: Assertion verifies the error message includes missing coord info.
        """
        self.processor.dataset = xr.Dataset({
            'dummy': xr.DataArray([1, 2, 3], dims=['nCells'])
        })
        
        with pytest.raises(ValueError) as ctx:
            self.processor.extract_spatial_coordinates()
        
        assert "Could not find spatial coordinates" in str(ctx.value)
    
    def test_extract_spatial_coordinates_alternative_names(self: "TestSpatialCoordinatesAndFiltering") -> None:
        """
        This test verifies that the `extract_spatial_coordinates` method can successfully extract longitude and latitude coordinates when they are provided under alternative variable names. The test provides a dataset with `longitude` and `latitude` variables instead of the standard `lonCell` and `latCell`, and asserts that the method returns coordinate arrays of the expected length, confirming that it can handle different naming conventions for spatial coordinates.

        Parameters:
            self (Any): Test case instance with processor fixture.

        Returns:
            None: Assertions verify returned arrays match provided alternative names.
        """
        self.processor.dataset = xr.Dataset({
            'longitude': xr.DataArray(np.array([0, 90]), dims=['nCells']),
            'latitude': xr.DataArray(np.array([0, 45]), dims=['nCells'])
        })
        
        lon, lat = self.processor.extract_spatial_coordinates()
        
        assert len(lon) == pytest.approx(2)
        assert len(lat) == pytest.approx(2)


class TestDatasetOperationsAndVariables:
    """ Tests for dataset operations and variable management. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestDatasetOperationsAndVariables", mock_mpas_mesh, mock_mpas_2d_data) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing dataset operations and variable management methods of the `MPASBaseProcessor`. It creates a temporary directory and grid file using the provided `mock_mpas_mesh` fixture, which contains real MPAS mesh data when available. The fixture initializes an instance of `MPASBaseProcessor` with the created grid file and assigns the provided `mock_mpas_2d_data` to an instance variable for use in the tests. After yielding control to the test methods, it performs cleanup by ensuring that the processor's dataset is reset to a known state using real MPAS data from the fixtures, allowing subsequent tests to operate on valid datasets.

        Parameters:
            mock_mpas_mesh: Fixture providing real or synthetic MPAS mesh data.
            mock_mpas_2d_data: Fixture providing real 2D diagnostic data.

        Returns:
            Generator[None, None, None]: Yields control to the test and performs cleanup after the test completes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)
        
        self.processor = MPASBaseProcessor(self.grid_file, verbose=False)
        self.mock_mpas_2d_data = mock_mpas_2d_data
            
        yield
        
        if 't2m' in self.mock_mpas_2d_data and 'surface_pressure' in self.mock_mpas_2d_data:
            self.processor.dataset = xr.Dataset({
                't2m': self.mock_mpas_2d_data['t2m'].isel(Time=0),
                'mslp': self.mock_mpas_2d_data['surface_pressure'].isel(Time=0)
            })
        elif 't2m' in self.mock_mpas_2d_data:
            self.processor.dataset = xr.Dataset({
                't2m': self.mock_mpas_2d_data['t2m'].isel(Time=0)
            })
            
        variables = self.processor.get_available_variables()
        
        assert 't2m' in variables
        assert len(variables) >= 1
    
    def test_get_available_variables_no_dataset(self: "TestDatasetOperationsAndVariables") -> None:
        """
        This test verifies that the `get_available_variables` method raises a `ValueError` when no dataset is loaded on the processor. The test ensures that callers receive a clear error indicating the missing dataset when attempting to retrieve available variables.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError) as ctx:
            self.processor.get_available_variables()

        assert "Dataset not loaded" in str(ctx.value)
    
    def test_get_time_info_no_dataset(self: "TestDatasetOperationsAndVariables") -> None:
        """
        This test verifies that the `get_time_info` method raises a `ValueError` when no dataset is loaded on the processor. The test calls the method with a sample index and checks the exception message for clarity. This ensures that users receive a clear error when attempting to access time information without a valid dataset loaded.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError) as ctx:
            self.processor.get_time_info(0)

        assert "Dataset not loaded" in str(ctx.value)


class TestHelperMethodsAndOutput:
    """ Tests for helper methods and verbose output generation. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestHelperMethodsAndOutput", mock_mpas_mesh, mock_mpas_2d_data) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing helper methods and verbose output generation of the `MPASBaseProcessor`. It creates a temporary directory and grid file using the provided `mock_mpas_mesh` fixture, which contains real MPAS mesh data when available. The fixture initializes an instance of `MPASBaseProcessor` with the created grid file and assigns the provided `mock_mpas_2d_data` to an instance variable for use in the tests. After yielding control to the test methods, it performs cleanup by ensuring that the processor's dataset is reset to a known state using real MPAS data from the fixtures, allowing subsequent tests to operate on valid datasets and verify the functionality of helper methods.

        Parameters:
            mock_mpas_mesh: Fixture providing real MPAS mesh with spatial vars.
            mock_mpas_2d_data: Fixture providing real 2D diagnostic data.

        Returns:
            Generator[None, None, None]: Yields control to the test and runs cleanup steps after the test completes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)
        
        self.processor = MPASBaseProcessor(self.grid_file, verbose=False)
        self.mock_mpas_2d_data = mock_mpas_2d_data
            
        yield
        
        combined_ds = xr.Dataset({
            't2m': self.mock_mpas_2d_data['t2m'].isel(Time=0)
        })
        
        result_ds = self.processor._add_spatial_coords_helper(
            combined_ds,
            dimensions_to_add=['nCells'],
            spatial_vars=['lonCell', 'latCell'],
            processor_type='2D'
        )
        
        assert 'lonCell' in result_ds.data_vars
        assert 'latCell' in result_ds.data_vars
        assert 'nCells' in result_ds.coords
    
    def test_add_spatial_coords_helper_verbose(self: "TestHelperMethodsAndOutput") -> None:
        """
        This test verifies that the `_add_spatial_coords_helper` method correctly adds spatial coordinate variables to the dataset and produces verbose output when `processor.verbose` is True. The test uses real MPAS 2D data from the fixture to create a combined dataset and calls the helper method to add spatial coordinates. It captures the standard output and asserts that the expected messages about loading the grid file and adding spatial coordinates are present, as well as confirming that the resulting dataset contains the new coordinate variables.

        Parameters:
            None

        Returns:
            None
        """
        self.processor.verbose = True

        combined_ds = xr.Dataset({
            't2m': self.mock_mpas_2d_data['t2m'].isel(Time=0)
        })
        
        from io import StringIO
        import sys

        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            result_ds = self.processor._add_spatial_coords_helper(
                combined_ds,
                dimensions_to_add=['nCells'],
                spatial_vars=['lonCell', 'latCell'],
                processor_type='2D'
            )

            output = captured_output.getvalue()
            
            assert "Grid file loaded" in output
            assert "Added spatial coordinate variable" in output
            assert 'lonCell' in result_ds.data_vars
            assert 'latCell' in result_ds.data_vars

        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_loading_success(self: "TestHelperMethodsAndOutput", mock_mpas_3d_data) -> None:
        """
        This test verifies that the `_print_loading_success` method produces the expected verbose output when `processor.verbose` is True. The test uses real MPAS 3D data from the fixture to create a dataset and calls the method to print loading success information. It captures the standard output and asserts that the messages about successfully loading files, time range, and vertical levels are present in the output, confirming that the method provides informative feedback about the loaded dataset.

        Parameters:
            mock_mpas_3d_data: Fixture providing real 3D MPAS data.

        Returns:
            None
        """
        self.processor.verbose = True
        
        if 'theta' in mock_mpas_3d_data:
            dataset = xr.Dataset({
                'theta': mock_mpas_3d_data['theta']
            })
        elif 'temperature' in mock_mpas_3d_data:
            dataset = xr.Dataset({
                'temperature': mock_mpas_3d_data['temperature']
            })
        else:
            pytest.skip("Required 3D variables not available in mock_mpas_3d_data")
        
        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            self.processor._print_loading_success(5, dataset, "xarray", "2D")
            output = captured_output.getvalue()
            
            assert "Successfully loaded 5 2d files" in output
            assert "Time range:" in output
            assert "Vertical levels:" in output
        finally:
            sys.stdout = sys.__stdout__

    def test_add_spatial_coords_no_matching_dimension(self: "TestHelperMethodsAndOutput", mock_mpas_mesh) -> None:
        """
        This test verifies that the `_add_spatial_coords_helper` method does not add spatial coordinate variables when the specified dimensions to add do not match any dimensions in the dataset. The test creates a dataset with a dimension that is not present in the grid file and calls the helper method to attempt to add spatial coordinates. It asserts that the resulting dataset remains unchanged (i.e., no new coordinate variables are added) and that the original variable is still present, confirming that the method correctly handles cases where expected dimensions are missing.

        Parameters:
            mock_mpas_mesh: Fixture providing real MPAS mesh data.

        Returns:
            None
        """
        processor = MPASBaseProcessor(GRID_FILE, verbose=False)
        
        if 'nEdges' in mock_mpas_mesh.dims:
            n_edges = len(mock_mpas_mesh['nEdges'])
            combined_ds = xr.Dataset({
                'var1': xr.DataArray(np.ones(n_edges), dims=['nEdges'])
            })
        else:
            combined_ds = xr.Dataset({
                'var1': xr.DataArray(np.ones(50), dims=['nEdges'])
            })
        
        result_ds = processor._add_spatial_coords_helper(
            combined_ds,
            dimensions_to_add=['nCells', 'nVertices'], 
            spatial_vars=['lonCell', 'latCell'],
            processor_type='2D'
        )
        
        assert 'var1' in result_ds.data_vars
    

    def test_print_loading_success_without_verbose(self: "TestHelperMethodsAndOutput", mock_mpas_2d_data) -> None:
        """
        This test verifies that the `_print_loading_success` method does not produce output when `processor.verbose` is False. The test uses real MPAS 2D data from the fixture to create a dataset and calls the method to print loading success information. It captures the standard output and asserts that it is empty, confirming that the method respects the verbose setting and does not emit messages when verbose mode is disabled.

        Parameters:
            mock_mpas_2d_data: Fixture providing real 2D diagnostic data.

        Returns:
            None
        """
        processor = MPASBaseProcessor(GRID_FILE, verbose=False)
        
        dataset = xr.Dataset({
            't2m': mock_mpas_2d_data['t2m']
        })
        
        processor._print_loading_success(5, dataset, "xarray", "2D")


class TestDataLoadingStrategies:
    """ Tests for various data loading strategies and fallback mechanisms. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestDataLoadingStrategies", mock_mpas_mesh, mock_mpas_2d_data) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing data loading strategies and fallback mechanisms of the `MPASBaseProcessor`. It creates a temporary directory and grid file using the provided `mock_mpas_mesh` fixture, which contains real MPAS mesh data when available. The fixture initializes an instance of `MPASBaseProcessor` with the created grid file. It also creates a test data file using the provided `mock_mpas_2d_data` fixture, which contains real 2D diagnostic data. After yielding control to the test methods, it performs cleanup by ensuring that the processor's dataset is reset to a known state using real MPAS data from the fixtures, allowing subsequent tests to operate on valid datasets and verify the functionality of data loading strategies.

        Parameters:
            mock_mpas_mesh: Fixture providing real or synthetic MPAS mesh data.
            mock_mpas_2d_data: Fixture providing real 2D diagnostic data.

        Returns:
            Generator[None, None, None]: Yields to the test and performs cleanup after completion.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)        
        self.processor = MPASBaseProcessor(self.grid_file, verbose=False)
        
        self.data_file = os.path.join(self.temp_dir, "data.nc")
        mock_mpas_2d_data.to_netcdf(self.data_file)
        
        yield
        
    def test_load_single_file_fallback_xarray(self: "TestDataLoadingStrategies") -> None:
        """
        This test verifies that the `_load_single_file_xarray` method successfully loads a single file using `xarray` and returns a valid dataset. The test uses a real MPAS 2D diagnostic file created in the fixture and asserts that the returned object is an instance of `xarray.Dataset`, confirming that the method can load data correctly when provided with a valid file path.

        Parameters:
            None

        Returns:
            None
        """
        result_ds = self.processor._load_single_file_xarray(self.data_file)
        assert isinstance(result_ds, xr.Dataset)
    
    def test_load_single_file_fallback_first_file(self: "TestDataLoadingStrategies") -> None:
        """
        This test verifies that the `_load_single_file_fallback` method correctly falls back to loading a single file using `xarray` when `uxarray` raises an exception. The test patches the `ux.open_dataset` method to raise an exception, simulating a failure in the `uxarray` loading path. It then calls the fallback method with a list containing a single valid file and asserts that the returned dataset is an instance of `xarray.Dataset` and that the data type indicates that `xarray` was used, confirming that the fallback mechanism works as intended when `uxarray` fails.

        Parameters:
            None

        Returns:
            None
        """
        result_ds, data_type = self.processor._load_single_file_fallback(
            "", [self.data_file]
        )
        
        assert data_type in ('xarray', 'uxarray')
        assert hasattr(result_ds, 'data_vars')

    def test_all_loading_fails_triggers_single_file_fallback(self: "TestDataLoadingStrategies") -> None:
        """
        This test verifies that when both the primary loading method (using `uxarray`) and the multi-file fallback method (using `xarray.open_mfdataset`) fail, the `_load_data` method correctly triggers the single-file fallback mechanism. The test patches both `ux.open_dataset` and `xr.open_mfdataset` to raise exceptions, simulating failures in both loading paths. It then calls the `_load_data` method and asserts that the resulting dataset is not None and that the output contains indications of using the single-file fallback, confirming that the method correctly handles multiple loading failures and falls back to a single file load.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        processor = MPAS3DProcessor(GRID_FILE, verbose=True)
        
        from io import StringIO
        import sys as sys_module
        captured_output = StringIO()
        sys_module.stdout = captured_output
        
        try:
            with patch('mpasdiag.processing.base.ux.open_dataset') as mock_ux:
                with patch('mpasdiag.processing.base.xr.open_mfdataset') as mock_mfd:
                    mock_ux.side_effect = Exception("UXarray failed")
                    mock_mfd.side_effect = Exception("Multi-file xarray failed")
                    
                    dataset, data_type = processor._load_data(
                        MPASOUT_DIR,
                        use_pure_xarray=False,
                        chunks={'Time': 1},
                        reference_file="",
                        data_type_label="3D"
                    )
                    
                    output = captured_output.getvalue()
                    
                    assert dataset is not None
                    assert 'single' in output.lower()
        finally:
            sys_module.stdout = sys_module.__stdout__

    def test_single_file_fallback_uxarray_fails_use_xarray(self: "TestDataLoadingStrategies") -> None:
        """
        This test verifies that the `_load_single_file_fallback` method correctly falls back to using `xarray` when `uxarray` raises an exception for single-file loading. The test patches the `ux.open_dataset` method to raise an exception, simulating a failure in the `uxarray` loading path. It then calls the fallback method and asserts that the returned `data_type` indicates that `xarray` was used and that the dataset is an `xarray.Dataset` instance.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        processor = MPAS3DProcessor(GRID_FILE, verbose=True)
        
        mpas_files = sorted([
            os.path.join(MPASOUT_DIR, f) 
            for f in os.listdir(MPASOUT_DIR) 
            if f.endswith('.nc')
        ])
        
        if mpas_files:
            dataset = processor._load_single_file_xarray(mpas_files[0])
            assert isinstance(dataset, xr.Dataset)
            
            dataset2 = processor._load_single_file_xarray(mpas_files[0])
            assert isinstance(dataset2, xr.Dataset)


class TestEdgeCasesAndErrorHandling:
    """ Tests for edge cases, boundary conditions, and error handling. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestEdgeCasesAndErrorHandling", mock_mpas_mesh) -> Generator[None, None, None]:
        """
        This fixture sets up a temporary environment for testing edge cases and error handling of the `MPASBaseProcessor`. It creates a temporary directory and grid file using the provided `mock_mpas_mesh` fixture, which contains real MPAS mesh data when available. The fixture initializes an instance of `MPASBaseProcessor` with the created grid file and an instance of `MPASVisualizer`. After yielding control to the test methods, it performs cleanup by calling the `normalize_longitude` method with a value that exceeds the valid range, ensuring that the method correctly normalizes it back into the expected range. This setup allows the tests to exercise edge cases related to geographic coordinate handling and ensures that cleanup steps validate the processor's behavior.

        Parameters:
            mock_mpas_mesh: Fixture providing real or synthetic MPAS mesh data.

        Returns:
            Generator[None, None, None]: Yields control to the test and runs cleanup checks after the test completes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)
        
        self.processor = MPASBaseProcessor(self.grid_file, verbose=False)
        self.visualizer = MPASVisualizer(verbose=False)
            
        yield
        
        result = self.processor.normalize_longitude(450.0)
        assert result == pytest.approx(90.0)
    
    def test_validate_geographic_extent_edge_values(self: "TestEdgeCasesAndErrorHandling") -> None:
        """
        This test verifies that the `validate_geographic_extent` method correctly identifies valid geographic extents that are at the edge of the acceptable longitude and latitude ranges. The test provides examples of geographic extents that include boundary values (e.g., -180, 180 for longitude and -90, 90 for latitude) and asserts that the method returns True, indicating that these extents are considered valid.

        Parameters:
            None

        Returns:
            None
        """
        assert self.processor.validate_geographic_extent((-180, 180, -90, 90))
        assert self.processor.validate_geographic_extent((0, 180, 0, 90))
    
    def test_add_spatial_coords_helper_missing_variables(self: "TestEdgeCasesAndErrorHandling", mock_mpas_2d_data) -> None:
        """
        This test verifies that the `_add_spatial_coords_helper` method does not add spatial coordinate variables when the specified spatial variables are not present in the grid file. The test uses real MPAS 2D data from the fixture to create a combined dataset and calls the helper method with spatial variable names that do not exist in the grid file. It asserts that the resulting dataset remains unchanged (i.e., no new coordinate variables are added) and that the original variable is still present, confirming that the method correctly handles cases where expected spatial variables are missing without causing errors.

        Parameters:
            mock_mpas_2d_data: Fixture providing real 2D diagnostic data.

        Returns:
            None
        """
        combined_ds = xr.Dataset({
            't2m': mock_mpas_2d_data['t2m'].isel(Time=0)
        })
        
        result_ds = self.processor._add_spatial_coords_helper(
            combined_ds,
            dimensions_to_add=['nCells'],
            spatial_vars=['nonexistent_var'],
            processor_type='2D'
        )
        
        assert 't2m' in result_ds.data_vars

    def test_time_series_empty_data(self: "TestEdgeCasesAndErrorHandling") -> None:
        """
        This test verifies that the `create_time_series_plot` method can handle empty time and value arrays without raising an exception. The test calls the method with empty lists for times and values and asserts that a valid figure object is still created, confirming that the method can gracefully handle cases where no data is available for plotting.

        Parameters:
            None

        Returns:
            None
        """
        times = []
        values = []
        
        fig, ax = self.visualizer.create_time_series_plot(times, values)
        assert fig is not None
    

    def test_histogram_single_value(self: "TestEdgeCasesAndErrorHandling") -> None:
        """
        This test verifies that the `create_histogram` method can handle an array with a single value without raising an exception. The test creates a numpy array containing a single value and calls the method to create a histogram. It asserts that a valid figure object is created, confirming that the method can gracefully handle cases where the input data has minimal variability.

        Parameters:
            None

        Returns:
            None
        """
        data = np.ones(100)        
        fig, ax = self.visualizer.create_histogram(data)
        assert fig is not None
    

    def test_wind_plot_all_nan_data(self: "TestEdgeCasesAndErrorHandling") -> None:
        """
        This test verifies that the `create_wind_plot` method raises a `ValueError` when all input wind component data (u and v) are NaN values. The test provides longitude and latitude arrays along with u and v arrays filled with NaN values, and asserts that the method raises an exception indicating that valid wind data is required for plotting. This ensures that the method correctly handles cases where the input data is invalid and provides informative error messages to the user.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([1, 2, 3])
        lat = np.array([1, 2, 3])
        u = np.array([np.nan, np.nan, np.nan])
        v = np.array([np.nan, np.nan, np.nan])
        
        with pytest.raises(ValueError):
            self.visualizer.create_wind_plot(
                lon, lat, u, v, 0, 5, 0, 5
            )


class TestTimeHandlingAndFormatting:
    """ Tests for time-related methods and datetime formatting. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestTimeHandlingAndFormatting", mpas_3d_processor) -> Generator[None, None, None]:
        """
        This fixture sets up the context for testing time handling and formatting methods of the `MPASBaseProcessor`. It checks if the `mpas_3d_processor` fixture is available, which provides a session-scoped processor with loaded 3D MPAS data. If the fixture is not available, it skips the tests that require real MPAS data. If available, it assigns the processor to an instance variable for use in the tests. After yielding control to the test methods, it performs cleanup by closing all matplotlib figures to ensure that no resources are left open after the tests complete.

        Parameters:
            mpas_3d_processor: Session-scoped fixture with loaded 3D data.

        Returns:
            Generator[None, None, None]: Yields the processor context to the test and performs cleanup after completion.
        """
        if mpas_3d_processor is None:
            pytest.skip("Real MPAS data not available for time handling tests")
        
        self.processor = mpas_3d_processor
        
        yield
        
        plt.close('all')
    
    def test_get_time_info_with_valid_index(self: "TestTimeHandlingAndFormatting") -> None:
        """
        This test verifies that the `get_time_info` method returns a non-empty string containing time information when provided with a valid positive time index. The test calls the method with an index of 0 and asserts that the returned value is a string and contains date/time information, confirming that the method correctly retrieves and formats time information from the dataset.

        Parameters:
            None

        Returns:
            None
        """
        time_info = self.processor.get_time_info(0, var_context="temperature")        
        assert isinstance(time_info, str)
        assert len(time_info) > 0
    
    def test_get_time_info_with_negative_index(self: "TestTimeHandlingAndFormatting") -> None:
        """
        This test verifies that the `get_time_info` method returns a non-empty string containing time information when provided with a valid negative time index. The test calls the method with an index of -1, which should retrieve information for the last time step, and asserts that the returned value is a string and contains date/time information, confirming that the method correctly handles negative indices to access time information from the dataset.

        Parameters:
            None

        Returns:
            None
        """
        time_info = self.processor.get_time_info(-1, var_context="pressure")
        
        assert isinstance(time_info, str)
        assert len(time_info) > 0
    
    def test_parse_file_datetimes_real_files(self: "TestTimeHandlingAndFormatting") -> None:
        """
        This test verifies that the `parse_file_datetimes` method can successfully parse datetime information from a list of real MPAS output files. The test collects all NetCDF files from the specified output directory, calls the method to parse their datetimes, and asserts that the returned value is a list with the same length as the input file list, confirming that the method can handle real file inputs and extract datetime information as expected.

        Parameters:
            None

        Returns:
            None
        """
        diag_files = sorted([
            os.path.join(MPASOUT_DIR, f) 
            for f in os.listdir(MPASOUT_DIR) 
            if f.endswith('.nc')
        ])
        
        datetimes = self.processor.parse_file_datetimes(diag_files)
        
        assert isinstance(datetimes, list)
        assert len(datetimes) == len(diag_files)
    
    def test_validate_time_parameters_valid_index(self: "TestTimeHandlingAndFormatting") -> None:
        """
        This test verifies that the `validate_time_parameters` method correctly validates a valid positive time index and returns the expected time dimension name, validated index, and time size. The test calls the method with an index of 0 and asserts that the returned values are of the correct types (string for dimension name, integer for index and size) and that the time size is greater than 0, confirming that the method can successfully validate time parameters against the dataset.

        Parameters:
            None

        Returns:
            None
        """
        time_dim_name, validated_index, time_size = self.processor.validate_time_parameters(0)
        
        assert isinstance(time_dim_name, str)
        assert isinstance(validated_index, int)
        assert isinstance(time_size, int)
        assert time_size > 0
    
    def test_validate_time_parameters_negative_index(self: "TestTimeHandlingAndFormatting") -> None:
        """
        This test verifies that the `validate_time_parameters` method correctly validates a valid negative time index and returns the expected time dimension name, validated index, and time size. The test calls the method with an index of -1, which should access the last time step, and asserts that the returned values are of the correct types (string for dimension name, integer for index and size) and that the time size is greater than 0, confirming that the method can handle negative indices to access time information from the dataset.

        Parameters:
            None

        Returns:
            None
        """
        time_dim_name, validated_index, time_size = self.processor.validate_time_parameters(-1)
        
        assert isinstance(time_dim_name, str)
        assert isinstance(validated_index, int)
        assert isinstance(time_size, int)
        assert time_size > 0


class TestVisualizerOperations:
    """ Tests for MPASVisualizer initialization, configuration, and operations. """
    
    def setup_method(self: "TestVisualizerOperations", mock_mpas_3d_data) -> None:
        """
        This method sets up the context for testing the operations of the `MPASVisualizer`. It initializes an instance of `MPASVisualizer` with specific parameters (e.g., `figsize`, `dpi`, `verbose`) and creates a temporary directory for any output files that may be generated during the tests. It also assigns the provided `mock_mpas_3d_data` to an instance variable for use in the tests. This setup allows the test methods to operate on a consistent visualizer instance and have access to mock data for testing visualization functionalities.

        Parameters:
            mock_mpas_3d_data: Fixture providing real 3D MPAS data for testing visualizations.

        Returns:
            None
        """
        self.visualizer = MPASVisualizer(figsize=(10, 14), dpi=100, verbose=False)
        self.temp_dir = tempfile.mkdtemp()
        self.mock_mpas_3d_data = mock_mpas_3d_data
    
    def teardown_method(self: "TestVisualizerOperations") -> None:
        """
        This method performs cleanup after each test method in the `TestVisualizerOperations` class. It checks if the temporary directory created during setup exists and removes it to ensure that no temporary files are left on the filesystem after the tests complete. Additionally, it closes all matplotlib figures to free up resources and prevent any interference with subsequent tests that may involve plotting.

        Parameters:
            None

        Returns:
            None
        """
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_init_default_parameters(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `MPASVisualizer` initializes with the correct default parameters when no explicit arguments are provided. The test creates an instance of `MPASVisualizer` without passing any parameters and asserts that the default values for `figsize`, `dpi`, and `verbose` are set as expected, and that the `fig` and `ax` attributes are initialized to None, confirming that the visualizer's constructor correctly assigns default values.

        Parameters:
            None

        Returns:
            None
        """
        visualizer = MPASVisualizer()
        
        assert visualizer.figsize == (pytest.approx(10), pytest.approx(14))
        assert visualizer.dpi == pytest.approx(100)
        assert visualizer.verbose
        assert visualizer.fig is None
        assert visualizer.ax is None
    
    def test_init_custom_parameters(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `MPASVisualizer` initializes with the correct custom parameters when explicit arguments are provided. The test creates an instance of `MPASVisualizer` with specific values for `figsize`, `dpi`, and `verbose`, and asserts that these values are correctly assigned to the instance attributes, confirming that the visualizer's constructor correctly handles custom initialization parameters.

        Parameters:
            None

        Returns:
            None
        """
        visualizer = MPASVisualizer(figsize=(12, 8), dpi=150, verbose=False)
        
        assert visualizer.figsize == (pytest.approx(12), pytest.approx(8))
        assert visualizer.dpi == pytest.approx(150)
        assert not visualizer.verbose
    
    def test_close_plot(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `close_plot` method correctly closes the current figure and resets the `fig` and `ax` attributes to None. The test first creates a figure and axis using `plt.subplots()`, assigns them to the visualizer's attributes, and asserts that they are not None. It then calls the `close_plot` method and asserts that both `fig` and `ax` are set back to None, confirming that the method effectively closes the plot and cleans up the visualizer's state.

        Parameters:
            None

        Returns:
            None
        """
        visualizer = MPASVisualizer()
        visualizer.fig, visualizer.ax = plt.subplots()
        
        assert visualizer.fig is not None
        
        visualizer.close_plot()
        
        assert visualizer.fig is None
        assert visualizer.ax is None
    
    def test_close_plot_when_none(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `close_plot` method can be called safely even when there is no existing figure or axis (i.e., when `fig` and `ax` are already None). The test creates an instance of `MPASVisualizer` without initializing a plot, calls the `close_plot` method, and asserts that no exceptions are raised and that the `fig` attribute remains None, confirming that the method can handle cases where there is nothing to close without causing errors.

        Parameters:
            None

        Returns:
            None
        """
        visualizer = MPASVisualizer()        
        visualizer.close_plot()
        
        assert visualizer.fig is None

    def test_add_regional_features_conus(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `add_regional_features` method correctly adds CONUS state boundaries when the plotting extent overlaps with the CONUS region. The test creates a GeoAxes with a PlateCarree projection, mocks the `add_feature` method to track calls, and calls `add_regional_features` with an extent that covers the CONUS area. It then asserts that the `add_feature` method was called, confirming that the method attempts to add state boundaries when the extent is appropriate.

        Parameters:
            None

        Returns:
            None
        """
        map_proj = ccrs.PlateCarree()
        self.visualizer.fig = plt.figure()
        self.visualizer.ax = self.visualizer.fig.add_subplot(111, projection=map_proj)
        
        assert self.visualizer.ax is not None
        assert isinstance(self.visualizer.ax, GeoAxes)
        assert hasattr(self.visualizer.ax, 'add_feature')
        self.visualizer.ax.add_feature = Mock()
        self.visualizer.add_regional_features(-125, -65, 25, 50)
        self.visualizer.ax.add_feature.assert_called()
    

    def test_add_regional_features_outside_conus(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `add_regional_features` method does not attempt to add CONUS state boundaries when the plotting extent does not overlap with the CONUS region. The test creates a GeoAxes with a PlateCarree projection, mocks the `add_feature` method to track calls, and calls `add_regional_features` with an extent that is outside of the CONUS area (e.g., over Europe). It then asserts that the `add_feature` method was not called, confirming that the method correctly identifies when state boundaries should not be added based on the geographic extent.

        Parameters:
            None

        Returns:
            None
        """
        map_proj = ccrs.PlateCarree()
        self.visualizer.fig = plt.figure()
        self.visualizer.ax = self.visualizer.fig.add_subplot(111, projection=map_proj)        
        self.visualizer.add_regional_features(-10, 40, 35, 60)


    def test_save_plot_multiple_formats(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `save_plot` method can save the current figure in multiple specified formats (e.g., PNG and PDF) without errors. The test creates a simple plot, calls `save_plot` with a base output path and a list of formats, and asserts that the expected output files are created for each format, confirming that the method correctly handles saving in multiple formats.

        Parameters:
            None

        Returns:
            None
        """
        output_path = os.path.join(self.temp_dir, "test_plot")
        
        self.visualizer.fig, self.visualizer.ax = plt.subplots()
        self.visualizer.ax.plot([1, 2, 3], [1, 2, 3])
        
        self.visualizer.save_plot(output_path, formats=['png', 'pdf'])
        
        assert os.path.exists(output_path + '.png')
        assert os.path.exists(output_path + '.pdf')
    

    def test_save_plot_no_figure(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `save_plot` method raises an `AssertionError` when there is no current figure to save (i.e., when `fig` is None). The test calls `save_plot` without creating a plot first and asserts that an exception is raised, confirming that the method correctly checks for the existence of a figure before attempting to save.

        Parameters:
            None

        Returns:
            None
        """
        output_path = os.path.join(self.temp_dir, "test_plot")
        
        with pytest.raises(AssertionError):
            self.visualizer.save_plot(output_path)

    def test_create_time_series_plot_numpy_datetime(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `create_time_series_plot` method can handle time inputs provided as numpy datetime64 objects. The test creates a list of numpy datetime64 objects and corresponding values, calls the method to create a time series plot, and asserts that a figure is created and that a line is plotted on the axes, confirming that the method can process numpy datetime inputs correctly.

        Parameters:
            None

        Returns:
            None
        """
        times = [datetime(2024, 1, 1, i) for i in range(3)]
        values = [1.0, 2.0, 3.0]
        
        fig, ax = self.visualizer.create_time_series_plot(times, values)
        
        assert fig is not None
        lines = ax.get_lines()
        assert len(lines) == pytest.approx(1)
    

    def test_create_time_series_custom_labels(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `create_time_series_plot` method correctly applies custom title and axis labels when provided. The test creates a list of datetime objects and corresponding values, calls the method with specific title, x-label, and y-label parameters, and asserts that the axes have the expected title and labels, confirming that the method correctly incorporates custom labeling into the plot.

        Parameters:
            None

        Returns:
            None
        """
        times = [datetime(2024, 1, 1, i) for i in range(3)]
        values = [100.0, 200.0, 300.0]
        
        fig, ax = self.visualizer.create_time_series_plot(
            times, values, 
            title="Custom Title",
            ylabel="Custom Y",
            xlabel="Custom X"
        )
        
        assert ax.get_title() == "Custom Title"
        assert ax.get_ylabel() == "Custom Y"
        assert ax.get_xlabel() == "Custom X"

    def test_create_histogram_with_nan(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `create_histogram` method can handle input data containing NaN and infinite values by filtering them out and still creating a valid histogram. The test creates a numpy array with a mix of finite, NaN, and infinite values, calls the method to create a histogram, and asserts that a figure is created and that the histogram is plotted using only the finite values, confirming that the method correctly handles non-finite data.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 4.0, -np.inf, 5.0])        
        fig, ax = self.visualizer.create_histogram(data, bins=10)
        
        assert fig is not None
        patches = ax.patches
        assert len(patches) > 0
    

    def test_create_histogram_log_scale(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `create_histogram` method correctly applies a logarithmic scale to the y-axis when the `log_scale` parameter is set to True. The test creates a numpy array with positive values suitable for log scaling, calls the method with `log_scale=True`, and asserts that the y-axis scale of the resulting plot is set to 'log', confirming that the method correctly configures the histogram for logarithmic scaling.

        Parameters:
            None

        Returns:
            None
        """
        data = np.logspace(0, 3, 1000)  # 1 to 1000        
        fig, ax = self.visualizer.create_histogram(data, log_scale=True)        
        assert ax.get_yscale() == 'log'
    

    def test_create_histogram_custom_bins(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `create_histogram` method correctly uses custom bin edges when the `bins` parameter is provided. The test creates a numpy array of data and defines specific bin edges, calls the method with these custom bins, and asserts that the histogram is created using the specified bins by checking the number of patches (bars) in the histogram, confirming that the method correctly applies custom binning to the histogram.

        Parameters:
            None

        Returns:
            None
        """
        data = np.linspace(-3, 3, 500)
        bins = np.linspace(-3, 3, 10)
        
        fig, ax = self.visualizer.create_histogram(data, bins=bins)
        
        assert fig is not None
    

    def test_create_histogram_empty_data(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `create_histogram` method can handle an empty input array without raising an exception. The test creates an empty numpy array, calls the method to create a histogram, and asserts that a figure is still created, confirming that the method can gracefully handle cases where there is no data to plot.

        Parameters:
            None

        Returns:
            None
        """
        data = np.array([np.nan, np.inf, -np.inf])
        fig, ax = self.visualizer.create_histogram(data)
        assert fig is not None

    def test_extract_2d_from_3d_by_value(self: "TestVisualizerOperations") -> None:
        """
        This test verifies that the `extract_2d_from_3d` method can successfully extract a 2D slice from a 3D data array based on a specified level value using the 'nearest' method. The test creates a mock 3D data array with a vertical dimension representing pressure levels and a horizontal dimension representing cells. It then calls the method to extract the 2D slice at a specific pressure level (e.g., 850 hPa) and asserts that the result is a 1D numpy array with the expected shape, confirming that the method can correctly identify and extract the nearest level from the 3D data.

        Parameters:
            None

        Returns:
            None
        """
        pressure_levels = np.linspace(1000.0, 100.0, 55)
        data_3d = xr.DataArray(
            np.linspace(200.0, 300.0, 55 * 100).reshape(55, 100),
            dims=['nVertLevels', 'nCells'],
            coords={'nVertLevels': pressure_levels}
        )
        result = self.visualizer.extract_2d_from_3d(data_3d, level_value=850.0, method='nearest')
        assert isinstance(result, np.ndarray)
        assert result.ndim == pytest.approx(1)
        assert result.shape == (100,)


class TestWindVisualization:
    """ Tests for wind plot creation and background overlay. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestWindVisualization") -> Generator[None, None, None]:
        """
        This fixture sets up the context for testing wind visualization functionalities of the `MPASVisualizer`. It initializes an instance of `MPASVisualizer` with verbose output disabled and creates sample wind data (longitude, latitude, u and v components) using a module-level helper function that generates realistic MPAS grid coordinates. The fixture yields control to the test methods, allowing them to use the initialized visualizer and sample data. After the tests complete, it performs cleanup by creating a wind plot using the sample data and asserting that a figure is produced and that the axis is a GeoAxes instance, confirming that the visualizer can successfully create a wind plot with the provided data.

        Parameters:
            None

        Returns:
            Generator[None, None, None]: Fixture generator for pytest.
        """
        self.visualizer = MPASVisualizer(verbose=False)        
        self.lon, self.lat, self.u, self.v = load_mpas_coords_from_processor(n=50)
            
        yield
        
        fig, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            plot_type='barbs'
        )
        
        assert fig is not None
        assert isinstance(ax, GeoAxes)
    
    def test_create_wind_plot_arrows(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `create_wind_plot` method can create a wind plot using arrows to represent wind vectors when `plot_type='arrows'`. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with the specified plot type, and asserts that a figure is created and that the axis is an instance of GeoAxes, confirming that the method can successfully generate a wind plot with arrow representations.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            plot_type='arrows'
        )
        
        assert fig is not None
        assert isinstance(ax, GeoAxes)
    
    def test_create_wind_plot_with_background(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `create_wind_plot` method can include a background wind-speed field when the `show_background` parameter is set to True. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with `show_background=True` and a specified colormap for the background, and asserts that a figure is created without errors, confirming that the method can successfully generate a wind plot with an overlaid background field.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            show_background=True,
            bg_colormap='plasma'
        )
        
        assert fig is not None
    
    def test_create_wind_plot_auto_subsample(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `create_wind_plot` method can automatically determine an appropriate subsampling factor when `subsample=0` is passed. The test uses a larger set of sample longitude, latitude, u, and v data initialized in the fixture, calls the method with `subsample=0`, and asserts that a figure is created without errors, confirming that the method can successfully compute an automatic subsampling factor to manage high-density wind data for plotting.

        Parameters:
            None

        Returns:
            None
        """
        lon, lat, u, v = load_mpas_coords_from_processor(n=1000)
        
        fig, ax = self.visualizer.create_wind_plot(
            lon, lat, u, v,
            -120, -80, 30, 50,
            subsample=0  
        )
        
        assert fig is not None
    
    def test_create_wind_plot_manual_subsample(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `create_wind_plot` method can accept a manual subsampling factor when the `subsample` parameter is set to a positive integer. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with a specified subsample value (e.g., 5), and asserts that a figure is created without errors, confirming that the method can successfully apply manual subsampling to manage wind data density for plotting.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            subsample=5
        )
        
        assert fig is not None
    
    def test_create_wind_plot_with_timestamp(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `create_wind_plot` method correctly incorporates a provided timestamp into the plot title when the `time_stamp` parameter is given. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with a specific datetime object for the timestamp, and asserts that the resulting plot title contains the formatted date and time information, confirming that the method can successfully include temporal context in the wind plot title.

        Parameters:
            None

        Returns:
            None
        """
        time_stamp = datetime(2024, 1, 15, 12, 0)
        
        fig, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            time_stamp=time_stamp
        )
        
        title = ax.get_title()
        assert '2024-01-15' in title
        assert '12:00 UTC' in title
    
    def test_create_wind_plot_custom_title(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `create_wind_plot` method correctly uses a custom title when the `title` parameter is provided. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with a specific string for the title, and asserts that the resulting plot title matches the provided custom title, confirming that the method can successfully apply user-defined titles to the wind plot.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            title="Custom Wind Title"
        )
        
        assert ax.get_title() == "Custom Wind Title"
    
    def test_create_wind_plot_custom_scale(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `create_wind_plot` method can accept a custom scaling factor for arrow plots when the `scale` parameter is provided. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with a specified scale value (e.g., 50.0) and `plot_type='arrows'`, and asserts that a figure is created without errors, confirming that the method can successfully apply custom scaling to the wind vectors in the plot.

        Parameters:
            None

        Returns:
            None
        """
        fig, ax = self.visualizer.create_wind_plot(
            self.lon, self.lat, self.u, self.v,
            -120, -80, 30, 50,
            plot_type='arrows',
            scale=50.0
        )
        
        assert fig is not None
    
    def test_create_wind_plot_invalid_type(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `create_wind_plot` method raises a `ValueError` when an invalid plot type is specified. The test uses the sample longitude, latitude, u, and v data initialized in the fixture, calls the method with an invalid `plot_type` value (e.g., 'invalid'), and asserts that a `ValueError` is raised with an appropriate message indicating that the plot type must be either 'barbs' or 'arrows', confirming that the method correctly validates the plot type parameter.

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(ValueError) as ctx:
            self.visualizer.create_wind_plot(
                self.lon, self.lat, self.u, self.v,
                -120, -80, 30, 50,
                plot_type='invalid'
            )
        
        assert "must be 'barbs' or 'arrows'" in str(ctx.value)
    
    def test_create_wind_plot_no_valid_data(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `create_wind_plot` method raises a `ValueError` when all wind data points are outside the specified plotting extent. The test creates sample longitude, latitude, u, and v data that are outside the defined extent for the plot, calls the method, and asserts that a `ValueError` is raised with a message indicating that there are no valid wind data points within the extent, confirming that the method correctly checks for valid data points before attempting to create the plot.

        Parameters:
            None

        Returns:
            None
        """
        lon = np.array([0, 10, 20])
        lat = np.array([0, 10, 20])
        u = np.array([1, 2, 3])
        v = np.array([1, 2, 3])
        
        with pytest.raises(ValueError) as ctx:
            self.visualizer.create_wind_plot(
                lon, lat, u, v,
                -120, -80, 30, 50 
            )
        
        assert "No valid wind data points" in str(ctx.value)

    def test_create_wind_background_no_axes(self: "TestWindVisualization") -> None:
        """
        This test verifies that the `_create_wind_background` method raises an `AssertionError` when there is no current axis (`ax`) to plot on. The test sets the visualizer's `ax` attribute to None, calls the method with sample data, and asserts that an exception is raised, confirming that the method correctly checks for the existence of an axis before attempting to create a background plot.

        Parameters:
            None

        Returns:
            None
        """
        self.visualizer.ax = None
        
        with pytest.raises(AssertionError):
            self.visualizer._create_wind_background(
                np.array([1, 2]), np.array([1, 2]), 
                np.array([1, 2]), 'viridis', ccrs.PlateCarree()
            )


class TestVerboseTruncationAndChunking:
    """ Tests for verbose file listing truncation and _apply_chunking exception. """

    @pytest.fixture(autouse=True)
    def setup_method(self, mock_mpas_mesh) -> Generator[None, None, None]:
        """
        This fixture sets up the context for testing the verbose file listing truncation and the behavior of the `_apply_chunking` method when it encounters an exception. It creates a temporary directory and a grid file from the mocked MPAS mesh, initializes an instance of `MPASBaseProcessor` with the grid file, and creates a data directory for test files. The fixture yields control to the test methods, allowing them to operate within this setup. After the tests complete, it cleans up by removing the temporary directory and all its contents, ensuring that no test artifacts remain on the filesystem.

        Parameters:
            mock_mpas_mesh: Mocked MPAS mesh object.

        Yields:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "grid.nc")
        mock_mpas_mesh.to_netcdf(self.grid_file)
        self.processor = MPASBaseProcessor(self.grid_file, verbose=True)
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_verbose_truncation_more_than_5_files(self) -> None:
        """
        This test verifies that the verbose file listing truncation works correctly when more than 5 files are found. It creates 8 test files in the data directory, captures the standard output, and calls the `_find_files_by_pattern` method. The test asserts that the output contains the message indicating that there are more files than displayed.

        Parameters:
            None

        Returns:
            None
        """
        for i in range(8):
            path = os.path.join(self.data_dir, f"diag.2024-01-{i+1:02d}_00.00.00.nc")
            xr.Dataset({'x': xr.DataArray([i])}).to_netcdf(path)

        from io import StringIO
        captured = StringIO()
        sys.stdout = captured
        try:
            files = self.processor._find_files_by_pattern(
                self.data_dir, "diag*.nc", "diagnostic files"
            )
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert len(files) == pytest.approx(8)
        assert "... and 3 more files" in output

    def test_apply_chunking_returns_original_on_failure(self) -> None:
        """
        This test verifies that the `_apply_chunking` method returns the original dataset when the `chunk()` method raises an exception. It creates a dataset, defines invalid chunking parameters, and asserts that the method returns the original dataset without raising an exception.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({'temp': xr.DataArray(np.ones(10))})
        bad_chunks = {'nonexistent_dim': 5}
        result = self.processor._apply_chunking(ds, bad_chunks)
        assert 'temp' in result

    def test_apply_chunking_none_returns_original(self) -> None:
        """
        This test verifies that the `_apply_chunking` method returns the original dataset unchanged when `chunks` is None. It creates a dataset, calls the method with `chunks=None`, and asserts that the returned dataset is identical to the original.

        Parameters:
            None

        Returns:
            None
        """
        ds = xr.Dataset({'temp': xr.DataArray(np.ones(10))})
        result = self.processor._apply_chunking(ds, None)
        xr.testing.assert_identical(result, ds)

    def test_attempt_primary_load_uxarray_success(self) -> None:
        """
        This test verifies that the `_attempt_primary_load` method successfully loads data using uxarray when `use_pure_xarray` is False and the data is compatible. It checks for the presence of real MPAS grid and mpasout data, attempts to load the data with `use_pure_xarray=False`, and asserts that the returned data type is 'uxarray', confirming that the method can correctly utilize uxarray for loading when appropriate.

        Parameters:
            None

        Returns:
            None
        """
        if not os.path.exists(GRID_FILE) or not os.path.isdir(MPASOUT_DIR):
            pytest.skip("Real MPAS grid or mpasout data not available")

        mpasout_files = sorted(glob.glob(os.path.join(MPASOUT_DIR, 'mpasout*.nc')))[:2]
        if not mpasout_files:
            pytest.skip("No mpasout files found")

        processor = MPASBaseProcessor(GRID_FILE, verbose=False)
        datetimes = [datetime(2024, 1, i + 1) for i in range(len(mpasout_files))]

        ds, dtype = processor._attempt_primary_load(
            mpasout_files, datetimes,
            open_chunks={'Time': 1},
            full_chunks=None,
            use_pure_xarray=False,
            data_type_label="test"
        )

        assert dtype == 'uxarray'

    def test_attempt_primary_load_pure_xarray(self) -> None:
        """
        This test verifies that the `_attempt_primary_load` method correctly returns an xarray.Dataset when `use_pure_xarray` is True. It creates two small data files, calls the method with `use_pure_xarray=True`, and asserts that the method returns an xarray.Dataset.

        Parameters:
            None

        Returns:
            None
        """
        datetimes = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        data_files = []
        for i, dt in enumerate(datetimes):
            fp = os.path.join(self.data_dir, f"mpasout.2024-01-0{i+1}.nc")
            xr.Dataset({
                'temp': xr.DataArray(np.ones((1, 10)), dims=['Time', 'nCells'])
            }).to_netcdf(fp)
            data_files.append(fp)

        ds, dtype = self.processor._attempt_primary_load(
            data_files, datetimes,
            open_chunks={'Time': 1},
            full_chunks=None,
            use_pure_xarray=True,
            data_type_label="test"
        )

        assert dtype == 'xarray'
        assert isinstance(ds, xr.Dataset)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
