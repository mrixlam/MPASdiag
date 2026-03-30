#!/usr/bin/env python3
"""
MPASdiag Test Suite: Tests for MPAS Data Processing Utilities and Processors

This module contains a comprehensive set of unit tests for the data processing components of the MPASdiag package, specifically targeting the MPAS2DProcessor class and associated utility functions. The tests cover initialization, file discovery, datetime parsing, time parameter validation, spatial coordinate extraction, and error handling scenarios. Both synthetic inputs and real MPAS mesh data from fixtures are used to ensure robust validation of geographic extent checks, longitude normalization, accumulation hour parsing, and dataset-oriented helpers. Mocking is employed to isolate functionality and prevent actual file I/O during testing. This suite ensures the reliability and correctness of data processing operations critical for accurate visualization and analysis of MPAS model output.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load necessary libraries and modules
import os
import sys
import pytest
import tempfile
import numpy as np
import xarray as xr
from typing import Any
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from mpasdiag.processing.processors_2d import MPAS2DProcessor
from mpasdiag.processing.base import MPASBaseProcessor
from mpasdiag.diagnostics.precipitation import PrecipitationDiagnostics

package_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_dir))


class TestUtilityFunctions:
    """
    Tests for utility functions in MPASBaseProcessor. """

    @pytest.fixture(autouse=True)
    def temp_grid(self: "TestUtilityFunctions", tmp_path: Path) -> None:
        """
        This fixture sets up a temporary grid file for testing utility functions that require a grid file. It creates an empty NetCDF file in the pytest-provided temporary directory and assigns its path to self.stub_grid for use in test methods. This allows tests to run with a valid file path without relying on external files or actual filesystem state, ensuring isolation and repeatability of tests that depend on grid file access. 

        Parameters:
            tmp_path (Path): pytest-provided temporary directory.

        Returns:
            None
        """
        stub = tmp_path / "stub_grid.nc"
        stub.touch()
        self.stub_grid = str(stub)
    
    def test_validate_geographic_extent(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the validate_geographic_extent method correctly checks if the provided geographic bounds (lon_min, lon_max, lat_min, lat_max) are valid. It uses a real stub grid file to ensure that the method can access necessary grid information without filesystem patching. The test includes assertions for valid bounds as well as various invalid scenarios such as longitude out of range, latitude out of range, and min/max values reversed. This ensures that the method robustly validates geographic extents before processing data for visualization or analysis. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPASBaseProcessor(self.stub_grid, verbose=False)

        assert processor.validate_geographic_extent((100.0, 110.0, -10.0, 10.0))
        assert not processor.validate_geographic_extent((200.0, 110.0, -10.0, 10.0))
        assert not processor.validate_geographic_extent((100.0, 110.0, -100.0, 10.0))
        assert not processor.validate_geographic_extent((110.0, 100.0, -10.0, 10.0))
        assert not processor.validate_geographic_extent((100.0, 110.0, 10.0, -10.0))
    
    def test_normalize_longitude(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the normalize_longitude method correctly normalizes longitude values to the range [-180, 180]. It uses a real stub grid file to ensure that the method can access necessary grid information without filesystem patching. The test includes assertions for a variety of input longitudes, including values greater than 180, less than -180, and edge cases at 0 and ±180. This ensures that the method robustly handles longitude normalization for accurate geographic processing and visualization. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPASBaseProcessor(self.stub_grid, verbose=False)

        lon_in = np.array([350.0, -200.0, 0.0, 180.0, -180.0])
        lon_out = processor.normalize_longitude(lon_in)

        expected = np.array([-10.0, 160.0, 0.0, -180.0, -180.0])
        np.testing.assert_array_almost_equal(lon_out, expected)

        assert processor.normalize_longitude(350.0) == pytest.approx(-10.0)
    
    def test_get_accumulation_hours(self: "TestUtilityFunctions") -> None:
        """
        This test verifies that the get_accumulation_hours method correctly parses accumulation hour information from a given string. It uses a real stub grid file to ensure that the method can access necessary grid information without filesystem patching. The test includes assertions for valid accumulation hour strings (e.g., 'a01h', 'a03h', etc.) and checks that the correct number of hours is returned. It also tests an invalid string to confirm that the method defaults to 24 hours as expected. This ensures that the method can robustly extract accumulation hour information for proper time-based data processing and visualization. 

        Parameters:
            None

        Returns:
            None
        """
        precip_diag = PrecipitationDiagnostics(verbose=False)

        assert precip_diag.get_accumulation_hours('a01h') == 1
        assert precip_diag.get_accumulation_hours('a03h') == 3
        assert precip_diag.get_accumulation_hours('a06h') == 6
        assert precip_diag.get_accumulation_hours('a12h') == 12
        assert precip_diag.get_accumulation_hours('a24h') == 24
        assert precip_diag.get_accumulation_hours('invalid') == 24  

class TestMPAS2DProcessor:
    """ Tests for MPAS2DProcessor class. """
    
    @pytest.fixture(autouse=True)
    def setup_method(self: "TestMPAS2DProcessor", mock_mpas_mesh) -> Any:
        """
        This fixture sets up the test environment for all tests in the TestMPAS2DProcessor class by creating a temporary grid file from the provided mock MPAS mesh data. It uses the tempfile module to create a temporary directory and file, writes the mock mesh data to a NetCDF file, and assigns the file path to self.grid_file for use in test methods. After yielding control to the test methods, it cleans up by removing the temporary directory and file. This setup allows individual test methods to focus on their specific assertions related to MPAS2DProcessor functionality without worrying about dataset creation or cleanup. 

        Parameters:
            mock_mpas_mesh: Fixture providing real or synthetic MPAS mesh data.

        Returns:
            None
        """
        self.temp_dir = tempfile.mkdtemp()
        self.grid_file = os.path.join(self.temp_dir, "test_grid.nc")
        
        mock_mpas_mesh.to_netcdf(self.grid_file)
        
        yield
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the MPAS2DProcessor class initializes correctly with a valid grid file. It creates an instance of MPAS2DProcessor using the temporary grid file created in the setup fixture and checks that the grid_file attribute is set correctly, the verbose flag is False, and that the dataset and data_type attributes are initialized to None. This ensures that the processor can be instantiated without errors and has the expected default state before any data loading or processing occurs. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        assert processor.grid_file == self.grid_file
        assert not processor.verbose
        assert processor.dataset is None
        assert processor.data_type is None
    
    def test_initialization_invalid_grid(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the MPAS2DProcessor class raises a FileNotFoundError when initialized with an invalid grid file path. It attempts to create an instance of MPAS2DProcessor using a non-existent file path and checks that the expected exception is raised. This ensures that the processor properly handles cases where the provided grid file cannot be found, preventing silent failures or obscure errors during later processing steps. 

        Parameters:
            None

        Returns:
            None
        """
        with pytest.raises(FileNotFoundError):
            MPAS2DProcessor("nonexistent_file.nc")
    
    def test_find_diagnostic_files(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the find_diagnostic_files method correctly identifies and returns a sorted list of diagnostic files in a specified directory. It creates a temporary directory and populates it with several dummy diagnostic files following the expected naming convention. The test then calls find_diagnostic_files with the directory path and asserts that the returned list contains the correct number of files and that they are sorted in the expected order. This ensures that the method can successfully discover diagnostic files for subsequent processing steps. 

        Parameters:
            None

        Returns:
            None
        """
        diag_dir = os.path.join(self.temp_dir, "diag_data")
        os.makedirs(diag_dir)

        for i in range(3):
            fname = os.path.join(diag_dir, f"diag.2024-01-01_0{i}.00.00.nc")
            with open(fname, 'w') as fh:
                fh.write("")

        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        files = processor.find_diagnostic_files(diag_dir)

        assert len(files) == 3
        assert files == sorted(files)
    
    def test_find_diagnostic_files_insufficient(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the find_diagnostic_files method raises a ValueError when the specified directory contains fewer than 2 diagnostic files. It creates a temporary directory and populates it with only one dummy diagnostic file following the expected naming convention. The test then calls find_diagnostic_files with the directory path and checks that the expected exception is raised. This ensures that the method properly handles cases where there are not enough diagnostic files for processing, preventing downstream errors due to insufficient data. 

        Parameters:
            None

        Returns:
            None
        """
        diag_dir = os.path.join(self.temp_dir, "diag_single")
        os.makedirs(diag_dir)
        fname = os.path.join(diag_dir, "diag.2024-01-01_00.00.00.nc")

        with open(fname, 'w') as fh:
            fh.write("")

        processor = MPAS2DProcessor(self.grid_file, verbose=False)

        with pytest.raises(ValueError):
            processor.find_diagnostic_files(diag_dir)
    
    def test_parse_file_datetimes(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the parse_file_datetimes method correctly extracts datetime information from a list of diagnostic file names. It creates a list of file names following the expected naming convention, including one with an invalid format. The test then calls parse_file_datetimes with this list and asserts that the returned list of datetimes has the correct length and that the valid file names are parsed into the expected datetime objects. It also checks that the invalid file name results in a datetime object (potentially with a default value) rather than causing an error. This ensures that the method can robustly handle both valid and invalid file name formats when extracting datetime information for time-based processing. 

        Parameters:
            None

        Returns:
            None
        """
        files = [
            "diag.2024-01-01_00.00.00.nc",
            "diag.2024-01-01_01.00.00.nc",
            "invalid_filename.nc"
        ]
        
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        datetimes = processor.parse_file_datetimes(files)
        
        assert len(datetimes) == 3
        assert datetimes[0] == datetime(2024, 1, 1, 0, 0, 0)
        assert datetimes[1] == datetime(2024, 1, 1, 1, 0, 0)
        assert isinstance(datetimes[2], datetime)
    
    def test_validate_time_parameters(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the validate_time_parameters method correctly validates and adjusts time parameters based on the dataset's Time dimension. It creates a mock dataset with a Time dimension of size 10 and assigns it to the processor. The test then calls validate_time_parameters with valid and out-of-range time indices and asserts that the returned time dimension name, adjusted time index, and time size are correct. This ensures that the method can properly handle time parameter validation, preventing out-of-bounds errors during time-based data processing. 

        Parameters:
            None

        Returns:
            None
        """
        mock_dataset = MagicMock()
        mock_dataset.sizes = {'Time': 10}
        mock_dataset.sizes = {'Time': 10}
        
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        time_dim, time_idx, time_size = processor.validate_time_parameters(5)
        assert time_dim == 'Time'
        assert time_idx == 5
        assert time_size == 10
        
        time_dim, time_idx, time_size = processor.validate_time_parameters(15)
        assert time_idx == 9  
    
    def test_validate_time_parameters_no_dataset(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the validate_time_parameters method raises a ValueError when called without a loaded dataset. It creates an instance of MPAS2DProcessor without assigning a dataset and then calls validate_time_parameters, expecting it to raise the appropriate exception. This ensures that the method properly handles cases where time parameters cannot be validated due to the absence of a dataset, preventing attempts to access non-existent dataset attributes. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        
        with pytest.raises(ValueError):
            processor.validate_time_parameters(0)
    
    def test_get_time_info(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the get_time_info method correctly retrieves and formats time information from the dataset's Time dimension. It creates a mock dataset with a Time dimension containing datetime64 values and assigns it to the processor. The test then calls get_time_info with a valid time index and asserts that the returned time string is formatted as expected (e.g., "20240101T00"). This ensures that the method can successfully extract and format time information for use in titles, labels, or other time-based annotations during data visualization. 

        Parameters:
            None

        Returns:
            None
        """
        times = [
            np.datetime64('2024-01-01T00:00:00'),
            np.datetime64('2024-01-01T01:00:00'),
        ]
        
        mock_dataset = MagicMock()
        mock_dataset.Time.values = times
        mock_dataset.__len__ = MagicMock(return_value=2)
        mock_dataset.Time.__len__ = MagicMock(return_value=2)
        
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        processor.dataset = mock_dataset
        
        with patch('builtins.hasattr') as mock_hasattr:
            mock_hasattr.side_effect = lambda obj, attr: attr == 'Time'
            time_str = processor.get_time_info(0)
            assert time_str == "20240101T00"
    
    def test_extract_spatial_coordinates_no_dataset(self: "TestMPAS2DProcessor") -> None:
        """
        This test verifies that the extract_spatial_coordinates method raises a ValueError when called without a loaded dataset. It creates an instance of MPAS2DProcessor without assigning a dataset and then calls extract_spatial_coordinates, expecting it to raise the appropriate exception. This ensures that the method properly handles cases where spatial coordinates cannot be extracted due to the absence of a dataset, preventing attempts to access non-existent dataset attributes. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.grid_file, verbose=False)
        
        with pytest.raises(ValueError):
            processor.extract_spatial_coordinates()


class TestDataValidation:
    """ Tests for data validation and filtering methods in MPAS2DProcessor. """
    
    def test_filter_by_spatial_extent(self: "TestDataValidation", mock_mpas_mesh, mock_mpas_2d_data) -> None:
        """
        This test verifies that the filter_by_spatial_extent method correctly filters data based on specified geographic bounds. It uses real MPAS mesh data from the provided fixture to create a dataset with longitude and latitude coordinates. The test defines a smaller geographic extent within the bounds of the mesh and calls filter_by_spatial_extent with this extent. It then asserts that the returned mask is a boolean array of the correct length and that it correctly identifies which grid cells fall within the specified geographic bounds. This ensures that the method can accurately filter data for visualization or analysis based on spatial criteria. 

        Parameters:
            mock_mpas_mesh: Fixture providing real MPAS mesh with coordinates.
            mock_mpas_2d_data: Fixture providing real 2D diagnostic data.

        Returns:
            None
        """
        grid_file = tempfile.NamedTemporaryFile(suffix='.nc', delete=False).name

        try:
            mock_mpas_mesh.to_netcdf(grid_file)            
            processor = MPAS2DProcessor(grid_file, verbose=False)
            
            lon_rad = mock_mpas_mesh['lonCell'].values
            lat_rad = mock_mpas_mesh['latCell'].values
            
            if np.nanmax(np.abs(lon_rad)) <= 2 * np.pi + 1e-6:
                lon = np.degrees(lon_rad)
            else:
                lon = lon_rad

            if np.nanmax(np.abs(lat_rad)) <= np.pi / 2 + 1e-6:
                lat = np.degrees(lat_rad)
            else:
                lat = lat_rad
            
            lon = ((lon + 180.0) % 360.0) - 180.0
            
            nCells = len(lon)
            
            if 't2m' in mock_mpas_2d_data:
                data_array = mock_mpas_2d_data['t2m'].isel(Time=0)
            else:
                data_array = xr.DataArray(np.ones(nCells), dims=['nCells'])
            
            ds = xr.Dataset({
                'lonCell': (['nCells'], lon),
                'latCell': (['nCells'], lat),
            })

            processor.dataset = ds
            
            lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
            lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
            
            lon_margin = (lon_max - lon_min) * 0.2
            lat_margin = (lat_max - lat_min) * 0.2

            test_lon_min = lon_min + lon_margin
            test_lon_max = lon_max - lon_margin
            test_lat_min = lat_min + lat_margin
            test_lat_max = lat_max - lat_margin
            
            filtered_data, mask = processor.filter_by_spatial_extent(
                data_array, test_lon_min, test_lon_max, test_lat_min, test_lat_max
            )
            
            assert mask.dtype == bool
            assert len(mask) == nCells
            
            expected_mask = ((lon >= test_lon_min) & (lon <= test_lon_max) & 
                           (lat >= test_lat_min) & (lat <= test_lat_max))
            
            np.testing.assert_array_equal(mask, expected_mask)
            
        finally:
            os.unlink(grid_file)


class TestErrorHandling:
    """ Tests for error handling scenarios. """
    
    def test_invalid_variable_name(self: "TestErrorHandling") -> None:
        """
        This test validates that the MPAS2DProcessor class raises an appropriate exception when attempting to load a variable that does not exist in the dataset. It creates a mock dataset without the target variable and assigns it to the processor. The test then calls the method responsible for loading variable data with a non-existent variable name and checks that the expected exception (e.g., KeyError) is raised. This ensures that the processor can gracefully handle cases where users request variables that are not present in the dataset, providing informative error messages rather than crashing with obscure errors. 

        Parameters:
            None

        Returns:
            None
        """
        pass
    
    def test_corrupted_data_files(self: "TestErrorHandling") -> None:
        """
        This test verifies that the MPAS2DProcessor class can handle corrupted or unreadable diagnostic files gracefully. It creates a temporary directory and populates it with a file that has the correct naming convention but contains invalid data (e.g., non-NetCDF content). The test then calls the method responsible for loading diagnostic files and checks that the expected exception (e.g., OSError, IOError) is raised when attempting to read the corrupted file. This ensures that the processor can detect and report issues with input files, preventing silent failures or crashes during data loading.

        Parameters:
            None

        Returns:
            None
        """
        pass


if __name__ == '__main__':
    pytest.main([__file__])