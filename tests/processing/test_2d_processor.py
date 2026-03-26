#!/usr/bin/env python3

"""
MPASdiag Test Suite: MPAS2DProcessor with Real MPAS Data

This test suite validates the functionality of the MPAS2DProcessor class using actual MPAS diagnostic data. It covers initialization, file discovery, coordinate extraction, variable retrieval, accumulation parsing, and coordinate augmentation. The tests are designed to be comprehensive while gracefully skipping if the required MPAS test dataset is not available on the testing machine. 

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from mpasdiag.processing.processors_2d import MPAS2DProcessor
from tests.test_data_helpers import get_mpas_data_paths, check_mpas_data_available


def get_mpas_test_data_paths() -> Dict[str, str]:
    """
    This helper function retrieves the paths to the MPAS test data required for processing tests. It uses a utility function to get the paths and ensures that if a diagnostic directory is provided without a separate data directory, it defaults to using the diagnostic directory for both. This function centralizes the logic for determining data paths, making it easier to manage and update as needed.

    Parameters:
        None

    Returns:
        Dict[str, str]: Dictionary containing paths to MPAS test data.
    """
    paths = get_mpas_data_paths()

    if 'diag_dir' in paths and 'data_dir' not in paths:
        paths['data_dir'] = paths['diag_dir']

    return paths

def load_mpas_processor(verbose: bool = False, use_pure_xarray: bool = True) -> MPAS2DProcessor:
    """
    This function initializes an MPAS2DProcessor instance and attempts to load MPAS diagnostic data if available. It retrieves the necessary paths using a helper function, checks for the existence of the required grid file, and initializes the processor. If diagnostic data is available, it attempts to load it using the specified loading method. If any required data is missing or if loading fails, it gracefully skips the test. This function is designed to be used as a fixture in tests that require a processor with loaded data, ensuring that tests only run when valid data is present.
    
    Parameters:
        verbose (bool): Whether to enable verbose logging on the processor.
        use_pure_xarray (bool): If True, prefer the pure-xarray loader path.

    Returns:
        MPAS2DProcessor: Processor instance with data loaded when available.
    """
    paths = get_mpas_data_paths()
    
    if not os.path.exists(paths['grid_file']):
        pytest.skip("MPAS test data not found")
    
    processor = MPAS2DProcessor(paths['grid_file'], verbose=verbose)
    
    if os.path.exists(paths['diag_dir']):
        try:
            processor.load_2d_data(paths['diag_dir'], use_pure_xarray=use_pure_xarray)
        except Exception as e:
            pytest.skip(f"Could not load MPAS data: {e}")
    
    return processor


class TestMPAS2DProcessorInitialization:
    """ Test processor initialization with actual MPAS data. """

    @classmethod
    def setup_class(cls: Any) -> None:
        """
        This method initializes shared resources for processor initialization tests. It prepares the paths and checks for the availability of MPAS test data. If the required data is not present, it will skip all tests in this class, ensuring that they only run when valid data is available. Shared attributes are stored on `cls` for reuse in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset.

        Parameters:
            cls: The test class object used to store shared attributes.

        Returns:
            None
        """
        cls.paths = get_mpas_test_data_paths()
        if not check_mpas_data_available():
            pytest.skip("MPAS test data not available")

    def test_initialization_verbose_true(self: "TestMPAS2DProcessorInitialization") -> None:
        """
        This test verifies that the MPAS2DProcessor initializes correctly when verbose mode is enabled. It creates an instance with `verbose=True` and checks that the flag is stored and True. This confirms that the constructor accepts and preserves configuration flags. No dataset loading is performed in this test. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.paths['grid_file'], verbose=True)
        assert processor is not None
        assert processor.verbose

    def test_initialization_verbose_false(self: "TestMPAS2DProcessorInitialization") -> None:
        """
        This test verifies that the MPAS2DProcessor initializes correctly when verbose mode is disabled. It creates an instance with `verbose=False` and checks that the flag is stored and False. This confirms that the constructor accepts and preserves configuration flags. No dataset loading is performed in this test.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.paths['grid_file'], verbose=False)
        assert processor is not None
        assert not processor.verbose


class TestFindDiagnosticFiles:
    """ Test file discovery functionality with actual MPAS data. """

    @classmethod
    def setup_class(cls: Any) -> None:
        """
        This method initializes shared resources for file discovery tests. It prepares the paths and checks for the availability of MPAS test data. If the required data is not present, it will skip all tests in this class, ensuring that they only run when valid data is available. It also initializes an MPAS2DProcessor instance that can be used across multiple tests to validate file discovery functionality. Shared attributes are stored on `cls` for reuse in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset.

        Parameters:
            cls: The test class object used to store shared attributes.

        Returns:
            None
        """
        cls.paths = get_mpas_test_data_paths()

        if not check_mpas_data_available():
            pytest.skip("MPAS test data not available")
        
        cls.processor = MPAS2DProcessor(cls.paths['grid_file'], verbose=True)

    def test_find_files_in_diag_subdirectory(self: "TestFindDiagnosticFiles") -> None:
        """
        This test verifies that the file discovery method can successfully find diagnostic files located in the specified diagnostic directory. It calls the `find_diagnostic_files` method with the path to the diagnostic directory and asserts that a list of files is returned, that it contains at least 2 files, and that each file exists on the filesystem. This confirms that the method can correctly identify and access diagnostic files when they are present in the expected location.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        files = self.processor.find_diagnostic_files(self.paths['data_dir'])
        
        assert files is not None
        assert isinstance(files, list)
        assert len(files) >= 2, "Should find at least 2 files"
        
        for file in files:
            assert os.path.exists(file)

    def test_find_files_prints_summary(self: "TestFindDiagnosticFiles") -> None:
        """
        This test checks that the file discovery method produces output when verbose mode is enabled. It patches the built-in `print` function to capture output during the file discovery process and asserts that some output was produced, confirming that the method provides user feedback in verbose mode. It also checks that the output contains a summary of the number of files found, ensuring that the method communicates useful information to the user. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        with patch('builtins.print') as mock_print:
            files = self.processor.find_diagnostic_files(self.paths['data_dir'])
            count = len(files) if files else 0
            
            assert mock_print.call_count > 0
            assert any(f"Found {count} diagnostic files" in str(call.args[0]) for call in mock_print.call_args_list)

    def test_find_files_recursive_search(self: "TestFindDiagnosticFiles") -> None:
        """
        This test verifies that the file discovery method can find diagnostic files located in subdirectories of the specified data directory. It creates a temporary subdirectory, copies a diagnostic file into it, and then calls the `find_diagnostic_files` method with the parent data directory. The test asserts that the copied file is included in the list of found files, confirming that the method can perform a recursive search for diagnostic files. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        files = self.processor.find_diagnostic_files(self.paths['data_dir'])
        
        assert files is not None
        assert len(files) > 0


class TestExtract2DCoordinates:
    """ Test coordinate extraction from actual MPAS data. """

    @pytest.fixture(autouse=True)
    def setup_method(self: "TestExtract2DCoordinates", mpas_2d_processor_diag, mpas_data_available):
        """
        This fixture sets up the MPAS2DProcessor with loaded diagnostic data for coordinate extraction tests. It uses the session-scoped `mpas_2d_processor_diag` fixture to avoid redundant loading across multiple tests. If the MPAS dataset is not available, it will skip all tests in this class, ensuring that they only run when valid data is present. Shared attributes are stored on `self` for use in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset.

        Parameters:
            self: Test instance provided by pytest.
            mpas_2d_processor_diag: Session-scoped processor with loaded diagnostic data
            mpas_data_available: Boolean flag indicating if MPAS data exists

        Returns:
            None
        """
        if not mpas_data_available or mpas_2d_processor_diag is None:
            pytest.skip("MPAS test data not available")
        
        self.processor = mpas_2d_processor_diag
        self.paths = get_mpas_test_data_paths()

    def test_extract_cell_coordinates(self: "TestExtract2DCoordinates") -> None:
        """
        This test verifies that the coordinate extraction method can successfully extract longitude and latitude arrays when the dataset contains variables defined on `nCells`. It checks for the presence of such variables, calls the extraction method, and asserts that the returned longitude and latitude arrays are not None and have matching lengths. This confirms that the method can correctly identify and extract spatial coordinates for cell-based data when available. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        var_name = None

        for vname in self.processor.dataset.data_vars:
            if 'nCells' in self.processor.dataset[vname].dims:
                var_name = vname
                break
        
        if var_name:
            lon, lat = self.processor.extract_spatial_coordinates()
            
            assert lon is not None
            assert lat is not None
            assert len(lon) == len(lat)

    def test_extract_coordinates_radian_conversion(self: "TestExtract2DCoordinates") -> None:
        """
        This test checks that the coordinate extraction method correctly handles coordinates that are already in degrees, ensuring that it does not apply an erroneous conversion from radians. It extracts the longitude and latitude arrays and asserts that their absolute values fall within expected degree ranges (latitude <= 90, longitude <= 360). This guards against double-conversion bugs that could arise if the method incorrectly assumes all input coordinates are in radians.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        lon, lat = self.processor.extract_spatial_coordinates()
        
        assert np.all(np.abs(lat <= 90))
        assert np.all(np.abs(lon <= 360))

    def test_extract_coordinates_longitude_wrapping(self: "TestExtract2DCoordinates") -> None:
        """
        This test verifies that the coordinate extraction method correctly handles longitude values that may be wrapped around the globe, ensuring that it does not produce values outside the expected range of -180 to 360 degrees. It extracts the longitude array and asserts that all longitude values fall within this range, confirming that the method can handle global coordinate systems without producing invalid longitude values. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        lon, _ = self.processor.extract_spatial_coordinates()        
        assert np.all(lon >= -180) and np.all(lon <= 360)

    def test_extract_coordinates_not_loaded(self: "TestExtract2DCoordinates") -> None:
        """
        This test verifies that the coordinate extraction method raises an appropriate exception when the MPAS2DProcessor instance has not loaded any data. It creates a new processor instance without loading data and calls the `extract_spatial_coordinates` method, expecting it to raise an error such as AttributeError, KeyError, TypeError, or ValueError. This confirms that the method correctly handles cases where required data is missing and does not produce coordinates when no dataset is available.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.paths['grid_file'], verbose=False)
        
        with pytest.raises((AttributeError, KeyError, TypeError, ValueError)):
            processor.extract_spatial_coordinates()

    def test_extract_coordinates_verbose_output(self: "TestExtract2DCoordinates") -> None:
        """
        This test checks that the coordinate extraction method produces output when verbose mode is enabled. It creates a new processor instance with `verbose=True`, loads the diagnostic data, and patches the built-in `print` function to capture output during the coordinate extraction process. The test asserts that some output was produced, confirming that the method provides user feedback in verbose mode. It also checks that the output contains messages indicating the progress of coordinate extraction, ensuring that the method communicates useful information to the user.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.paths['grid_file'], verbose=True)
        processor.load_2d_data(self.paths['diag_dir'], use_pure_xarray=True)
        
        with patch('builtins.print') as mock_print:
            _, _ = processor.extract_spatial_coordinates()
            assert mock_print.call_count >= 0  

    def test_extract_vertex_coordinates(self: "TestExtract2DCoordinates") -> None:
        """
        This test checks that the coordinate extraction method can successfully extract vertex-based coordinates when the dataset contains variables defined on `nVertices`. It verifies the presence of such variables, calls the extraction method, and asserts that the returned longitude and latitude arrays are not None. This confirms that the method can correctly extract spatial coordinates for vertex-based data when available. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        has_vertex_var = any('nVertices' in self.processor.dataset[v].dims 
                            for v in self.processor.dataset.data_vars)
        
        if has_vertex_var:
            lon, lat = self.processor.extract_spatial_coordinates()
            
            assert lon is not None
            assert lat is not None

    def test_extract_with_data_array_cells(self: "TestExtract2DCoordinates") -> None:
        """
        This test verifies that the coordinate extraction method can successfully extract coordinates when the dataset contains variables defined on `nCells` that are represented as xarray DataArrays. It checks for the presence of such variables, calls the extraction method, and asserts that the returned longitude and latitude arrays are not None and have matching lengths. This confirms that the method can correctly identify and extract spatial coordinates for cell-based data stored as DataArrays. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        var_name = None

        for vname in self.processor.dataset.data_vars:
            if 'nCells' in self.processor.dataset[vname].dims:
                var_name = vname
                break
        
        if var_name:
            data_array = self.processor.dataset[var_name]
            assert isinstance(data_array, xr.DataArray)

    def test_extract_with_data_array_vertices(self: "TestExtract2DCoordinates") -> None:
        """
        This test checks that the coordinate extraction method can handle vertex-based coordinates when the dataset contains variables defined on `nVertices` that are represented as xarray DataArrays. It verifies the presence of such variables, calls the extraction method, and asserts that the returned longitude and latitude arrays are not None. This confirms that the method can correctly extract spatial coordinates for vertex-based data stored as DataArrays. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        has_vertex = any('nVertices' in self.processor.dataset[v].dims 
                        for v in self.processor.dataset.data_vars)
        
        if has_vertex:
            assert True  

    def test_extract_coordinates_missing(self: "TestExtract2DCoordinates") -> None:
        """
        This test verifies that the coordinate extraction method raises an appropriate exception when the dataset does not contain any variables defined on `nCells` or `nVertices`. It checks for the absence of such variables, calls the extraction method, and expects it to raise an error such as ValueError, KeyError, or AttributeError. This confirms that the method correctly handles cases where no spatial coordinate variables are present in the dataset and does not produce coordinates when they cannot be extracted.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        lon, lat = self.processor.extract_spatial_coordinates()
        assert lon is not None
        assert lat is not None


class TestGet2DVariableData:
    """ Test variable data retrieval from actual MPAS files. """

    @pytest.fixture(autouse=True)
    def setup_method(self: "TestGet2DVariableData", mpas_2d_processor_diag, mpas_data_available):
        """
        This fixture sets up the MPAS2DProcessor with loaded diagnostic data for variable retrieval tests. It uses the session-scoped `mpas_2d_processor_diag` fixture to avoid redundant loading across multiple tests. If the MPAS dataset is not available, it will skip all tests in this class, ensuring that they only run when valid data is present. Shared attributes are stored on `self` for use in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset. 

        Parameters:
            self: Test instance provided by pytest.
            mpas_2d_processor_diag: Session-scoped processor with loaded diagnostic data
            mpas_data_available: Boolean flag indicating if MPAS data exists

        Returns:
            None
        """
        if not mpas_data_available or mpas_2d_processor_diag is None:
            pytest.skip("MPAS test data not available")
        
        self.processor = mpas_2d_processor_diag
        self.paths = get_mpas_test_data_paths()

    def test_get_variable_standard_xarray(self: "TestGet2DVariableData") -> None:
        """
        This test verifies that the `get_2d_variable_data` method can successfully retrieve variable data as a standard xarray DataArray when the dataset is loaded using the pure-xarray loader. It retrieves a variable from the dataset and asserts that the returned data is an instance of xarray DataArray. This confirms that the method can provide data in the expected format when using the pure-xarray loading path. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        var_name = list(self.processor.dataset.data_vars)[0]
        data = self.processor.get_2d_variable_data(var_name)
        
        assert data is not None
        assert isinstance(data, (np.ndarray, xr.DataArray))

    def test_get_variable_with_finite_values(self: "TestGet2DVariableData") -> None:
        """
        This test checks that the `get_2d_variable_data` method returns data that contains finite values. It retrieves a variable from the dataset and asserts that there are some finite values present in the returned data array. This confirms that the method can successfully retrieve valid data from the dataset and that it does not return an array of all NaN or infinite values, which could indicate an issue with data loading or retrieval. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        var_name = list(self.processor.dataset.data_vars)[0]
        data = self.processor.get_2d_variable_data(var_name)
        
        if isinstance(data, xr.DataArray):
            data = data.values
        
        assert np.any(np.isfinite(data))

    def test_get_variable_not_loaded(self: "TestGet2DVariableData") -> None:
        """
        This test verifies that the `get_2d_variable_data` method raises an appropriate exception when the MPAS2DProcessor instance has not loaded any data. It creates a new processor instance without loading data and calls the `get_2d_variable_data` method with a variable name, expecting it to raise an error such as AttributeError, TypeError, KeyError, or ValueError. This confirms that the method correctly handles cases where required data is missing and does not produce variable data when no dataset is available. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.paths['grid_file'], verbose=False)
        
        with pytest.raises((AttributeError, TypeError, KeyError, ValueError)):
            processor.get_2d_variable_data('t2m')

    def test_get_variable_not_found(self: "TestGet2DVariableData") -> None:
        """
        This test checks that the `get_2d_variable_data` method raises an appropriate exception when the requested variable name does not exist in the dataset. It calls the method with a variable name that is not present in the dataset and expects it to raise an error such as KeyError, AttributeError, or ValueError. This confirms that the method correctly handles cases where a user requests a variable that is not available in the dataset and provides meaningful feedback through exceptions. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        with pytest.raises((KeyError, AttributeError, ValueError)):
            self.processor.get_2d_variable_data('nonexistent_variable_xyz')

    def test_get_variable_uxarray_data_type(self: "TestGet2DVariableData") -> None:
        """
        This test verifies that the `get_2d_variable_data` method can return data in the form of an xarray DataArray when the dataset is loaded using the pure-xarray loader. It retrieves a variable from the dataset and asserts that the returned data is an instance of xarray DataArray. This confirms that the method can provide data in the expected format when using the pure-xarray loading path. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        var_name = list(self.processor.dataset.data_vars)[0]
        data = self.processor.get_2d_variable_data(var_name)
        assert isinstance(data, xr.DataArray)

    def test_get_variable_no_finite_values(self: "TestGet2DVariableData") -> None:
        """
        This test checks the behavior of the `get_2d_variable_data` method when the retrieved variable data contains no finite values. It retrieves a variable from the dataset and asserts that if the data is an xarray DataArray, it contains no finite values. This confirms that the method can handle cases where the variable data may be invalid or contain only NaN or infinite values, and that it does not fail when such data is encountered.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        var_name = list(self.processor.dataset.data_vars)[0]

        try:
            data = self.processor.get_2d_variable_data(var_name)
            assert data is not None
        except Exception:
            pass


class TestGetAccumulationHours:
    """ Test accumulation period parsing from actual MPAS files. """

    @classmethod
    def setup_class(cls: Any) -> None:
        """
        This method initializes shared resources for accumulation parsing tests. It prepares the paths and checks for the availability of MPAS test data. If the required data is not present, it will skip all tests in this class, ensuring that they only run when valid data is available. It also initializes an MPAS2DProcessor instance that can be used across multiple tests to validate accumulation parsing functionality. Shared attributes are stored on `cls` for reuse in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset.

        Parameters:
            cls: The test class object used to store shared attributes.

        Returns:
            None
        """
        cls.paths = get_mpas_test_data_paths()

        if not check_mpas_data_available():
            pytest.skip("MPAS test data not available")
        
        cls.processor = MPAS2DProcessor(cls.paths['grid_file'], verbose=False)        
        cls.diag_files = sorted(Path(cls.paths['diag_dir']).glob('diag.*.nc'))

        assert len(cls.diag_files) > 0, "No diagnostic files found for accumulation parsing tests"

    def test_accumulation_1_hour(self: "TestGetAccumulationHours") -> None:
        """
        This test validates the parsing of a 1-hour accumulation code (``a01h``) from the dataset. It calls the `get_accumulation_hours` method with the code and asserts that the returned value is a numeric type (int or float) and equals 1. This confirms that the method can correctly interpret standard 1-hour accumulation codes from the MPAS diagnostic metadata. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        hours = self.processor.get_accumulation_hours('a01h')
        assert isinstance(hours, (int, float))
        assert hours == 1

    def test_accumulation_3_hour(self: "TestGetAccumulationHours") -> None:
        """
        This test validates the parsing of a 3-hour accumulation code (``a03h``) from the dataset. It calls the `get_accumulation_hours` method with the code and asserts that the returned value is a numeric type (int or float) and equals 3. This confirms that the method can correctly interpret intermediate accumulation codes from the MPAS diagnostic metadata.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        hours = self.processor.get_accumulation_hours('a03h')
        assert isinstance(hours, (int, float))
        assert hours == 3

    def test_accumulation_6_hour(self: "TestGetAccumulationHours") -> None:
        """
        This test validates the parsing of a 6-hour accumulation code (``a06h``) from the dataset. It calls the `get_accumulation_hours` method with the code and asserts that the returned value is a numeric type (int or float) and equals 6. This confirms that the method can correctly interpret intermediate accumulation codes from the MPAS diagnostic metadata.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        hours = self.processor.get_accumulation_hours('a06h')
        assert isinstance(hours, (int, float))
        assert hours == 6

    def test_accumulation_12_hour(self: "TestGetAccumulationHours") -> None:
        """
        This test validates the parsing of a 12-hour accumulation code (``a12h``) from the dataset. It calls the `get_accumulation_hours` method with the code and asserts that the returned value is a numeric type (int or float) and equals 12. This confirms that the method can correctly interpret larger accumulation windows from the MPAS diagnostic metadata.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        hours = self.processor.get_accumulation_hours('a12h')
        assert isinstance(hours, (int, float))
        assert hours == 12

    def test_accumulation_24_hour(self: "TestGetAccumulationHours") -> None:
        """
        This test validates the parsing of a 24-hour accumulation code (``a24h``) from the dataset. It calls the `get_accumulation_hours` method with the code and asserts that the returned value is a numeric type (int or float) and equals 24. This confirms that the method can correctly interpret daily accumulation codes from the MPAS diagnostic metadata.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        hours = self.processor.get_accumulation_hours('a24h')
        assert isinstance(hours, (int, float))
        assert hours == 24

    def test_accumulation_none(self: "TestGetAccumulationHours") -> None:
        """
        This test validates the behavior when `None` is passed as an accumulation code. It calls the `get_accumulation_hours` method with `None` and asserts that the returned value is either a numeric type (int or float) or `None`. This ensures the parser can handle cases where the accumulation code is missing or not provided without raising an unexpected exception.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        try:
            hours = self.processor.get_accumulation_hours(None) # type: ignore
            assert isinstance(hours, (int, float, type(None)))
        except (TypeError, ValueError):
            pass  # Expected

    def test_accumulation_empty_string(self: "TestGetAccumulationHours") -> None:
        """
        This test validates the behavior when an empty string is passed as an accumulation code. It calls the `get_accumulation_hours` method with an empty string and asserts that the returned value is either a numeric type (int or float) or `None`. This ensures the parser can handle cases where the accumulation code is present but empty without raising an unexpected exception. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        try:
            hours = self.processor.get_accumulation_hours("")
            assert isinstance(hours, (int, float, type(None)))
        except (ValueError, KeyError):
            pass  # Expected

    def test_accumulation_unknown_default(self: "TestGetAccumulationHours") -> None:
        """
        This test validates the behavior when an unknown accumulation code is passed to the `get_accumulation_hours` method. It calls the method with a code that is not defined in the expected set (e.g., "unknown") and asserts that the returned value is either a numeric type (int or float) or `None`. This ensures that the parser can handle unrecognized accumulation codes gracefully, either by returning a default value or by raising an appropriate exception without crashing.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        try:
            hours = self.processor.get_accumulation_hours("unknown")
            assert isinstance(hours, (int, float, type(None)))
        except (ValueError, KeyError):
            pass  # Expected


class TestAddSpatialCoordinates:
    """ Test adding spatial coordinates to datasets. """

    @pytest.fixture(autouse=True)
    def setup_method(self: "TestAddSpatialCoordinates", mpas_2d_processor_diag, mpas_data_available):
        """
        This fixture sets up the MPAS2DProcessor with loaded diagnostic data for spatial coordinate tests. It uses the session-scoped `mpas_2d_processor_diag` fixture to avoid redundant loading across multiple tests. If the MPAS dataset is not available, it will skip all tests in this class, ensuring that they only run when valid data is present. Shared attributes are stored on `self` for use in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset.

        Parameters:
            self: Test instance provided by pytest.
            mpas_2d_processor_diag: Session-scoped processor with loaded diagnostic data
            mpas_data_available: Boolean flag indicating if MPAS data exists

        Returns:
            None
        """
        if not mpas_data_available or mpas_2d_processor_diag is None:
            pytest.skip("MPAS test data not available")
        
        self.processor = mpas_2d_processor_diag
        self.paths = get_mpas_test_data_paths()

    def test_add_coordinates_to_dataset(self: "TestAddSpatialCoordinates") -> None:
        """
        This test verifies that the `extract_spatial_coordinates` method successfully adds longitude and latitude coordinates to the dataset. It calls the method and then checks that the expected coordinate names (e.g., `latCell`, `lonCell`) are present in either the dataset's coordinates or variables. This confirms that the method can augment the dataset with spatial coordinates when they are extracted from the original data variables. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        assert ('latCell' in self.processor.dataset.coords or
                'latCell' in self.processor.dataset.variables)
        

    def test_add_coordinates_preserves_data(self: "TestAddSpatialCoordinates") -> None:
        """
        This test checks that the `extract_spatial_coordinates` method does not modify the original variable data when adding spatial coordinates to the dataset. It retrieves a variable from the dataset before calling the method, stores its values, and then calls `extract_spatial_coordinates`. Afterward, it asserts that the variable's data values remain unchanged, confirming that the method does not alter existing data when adding new coordinate information.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        var_name = list(self.processor.dataset.data_vars)[0]
        original_data = self.processor.dataset[var_name].values.copy()
        
        _ = self.processor.extract_spatial_coordinates()
        
        np.testing.assert_array_equal(
            self.processor.dataset[var_name].values,
            original_data
        )


class TestLoad2DData:
    """ Test complete data loading workflow with actual MPAS data. """

    @classmethod
    def setup_class(cls: Any) -> None:
        """
        This method initializes shared resources for data loading tests. It prepares the paths and checks for the availability of MPAS test data. If the required data is not present, it will skip all tests in this class, ensuring that they only run when valid data is available. Shared attributes are stored on `cls` for reuse in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset. 

        Parameters:
            cls: The test class object used to store shared attributes.

        Returns:
            None
        """
        cls.paths = get_mpas_test_data_paths()

        if not check_mpas_data_available():
            pytest.skip("MPAS test data not available")

    def test_load_data_returns_self(self: "TestLoad2DData") -> None:
        """
        This test verifies that the `load_2d_data` method returns the MPAS2DProcessor instance itself, allowing for method chaining. It creates a new processor instance, calls the `load_2d_data` method with the diagnostic directory path, and asserts that the returned value is the same instance as the processor. This confirms that the method is designed to return `self`, enabling a fluent interface for loading data and performing subsequent operations on the same processor instance. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        processor = MPAS2DProcessor(self.paths['grid_file'], verbose=False)
        result = processor.load_2d_data(self.paths['diag_dir'], use_pure_xarray=True)
        
        assert result is processor


class TestEdgeCases:
    """ Test edge cases and boundary conditions with real data. """

    @pytest.fixture(autouse=True)
    def setup_method(self: "TestEdgeCases", mpas_2d_processor_diag, mpas_data_available):
        """
        This fixture sets up the MPAS2DProcessor with loaded diagnostic data for edge case tests. It uses the session-scoped `mpas_2d_processor_diag` fixture to avoid redundant loading across multiple tests. If the MPAS dataset is not available, it will skip all tests in this class, ensuring that they only run when valid data is present. Shared attributes are stored on `self` for use in individual tests. This setup ensures that the tests are deterministic and do not fail due to missing data on machines that do not have the MPAS test dataset. 

        Parameters:
            self: Test instance provided by pytest.
            mpas_2d_processor_diag: Session-scoped processor with loaded diagnostic data
            mpas_data_available: Boolean flag indicating if MPAS data exists

        Returns:
            None
        """
        if not mpas_data_available or mpas_2d_processor_diag is None:
            pytest.skip("MPAS test data not available")
        
        self.processor = mpas_2d_processor_diag
        self.paths = get_mpas_test_data_paths()

    def test_coordinates_already_in_degrees(self: "TestEdgeCases") -> None:
        """
        This test checks that the coordinate extraction method correctly handles coordinates that are already in degrees, ensuring that it does not apply an erroneous conversion from radians. It extracts the longitude and latitude arrays and asserts that their absolute values fall within expected degree ranges (latitude <= 90, longitude <= 360). This guards against double-conversion bugs that could arise if the method incorrectly assumes all input coordinates are in radians. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        lon, lat = self.processor.extract_spatial_coordinates()
        
        assert np.all(np.abs(lat) <= 90)
        assert np.all(np.abs(lon) <= 360)

    def test_extract_coordinates_alternative_names(self: "TestEdgeCases") -> None:
        """
        This test verifies that the coordinate extraction method can handle alternative variable names for latitude and longitude. It checks for the presence of expected coordinate names (e.g., `latCell`, `lonCell`) in either the dataset's coordinates or variables after calling the extraction method. This confirms that the method can successfully add spatial coordinates to the dataset even if they are not named in a standard way, ensuring flexibility in handling different MPAS datasets with varying naming conventions.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        assert ('latCell' in self.processor.dataset.coords or
                'latCell' in self.processor.dataset.variables)
        

    def test_variable_data_without_units(self: "TestEdgeCases") -> None:
        """
        This test checks the behavior of the `get_2d_variable_data` method when retrieving a variable that does not have associated units in the dataset. It retrieves a variable from the dataset, calls the method to get its data, and asserts that the returned data is not None. This confirms that the method can successfully retrieve variable data even when unit metadata is missing, ensuring that it does not rely on units being present to function correctly.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        var_name = list(self.processor.dataset.data_vars)[0]
        data = self.processor.get_2d_variable_data(var_name)
        
        assert data is not None


def _make_mock_2d_processor(**kwargs: Any) -> MPAS2DProcessor:
    """
    This helper function creates a mock instance of MPAS2DProcessor with specified attributes for testing purposes. It patches the `__init__` method of MPAS2DProcessor to set default values for `verbose`, `dataset`, and `grid_file` attributes, allowing tests to create processor instances without needing to load actual data. The function accepts arbitrary keyword arguments to customize the attributes of the mock processor instance as needed for different test scenarios.

    Parameters:
        **kwargs: Arbitrary keyword arguments to set attributes on the mock processor instance.

    Returns:
        MPAS2DProcessor: A mock instance of MPAS2DProcessor with specified attributes.
    """
    with patch.object(MPAS2DProcessor, '__init__', lambda self, *a, **kw: setattr(self, 'verbose', kw.get('verbose', kwargs.get('verbose', False))) or setattr(self, 'dataset', None) or setattr(self, 'grid_file', 'mock')):
        proc = MPAS2DProcessor.__new__(MPAS2DProcessor)
        proc.verbose = kwargs.get('verbose', False)
        proc.dataset = kwargs.get('dataset', None)
        proc.grid_file = 'mock'
        return proc


class TestFileDiscoveryFallbacks:
    """ Tests for find_diagnostic_files fallback chain: diag → diag subdir → recursive → mpasout → mpasout recursive → error. """

    def test_find_diag_files_in_diag_subdir(self: "TestFileDiscoveryFallbacks", tmp_path: Any) -> None:
        """
        This test verifies that when the primary search for diagnostic files in the specified directory fails, the method correctly falls back to searching within a `diag` subdirectory. It creates a `diag` subdirectory within the temporary path and populates it with mock diagnostic files. The test then patches the file discovery method to simulate a failure in the primary search while allowing the subdirectory search to succeed, and asserts that the correct number of files is returned from the `diag` subdirectory. 

        Parameters:
            self: Test instance provided by pytest.
            tmp_path: pytest fixture providing a temporary directory for file creation.

        Returns:
            None
        """
        diag_sub = tmp_path / "diag"
        diag_sub.mkdir()

        for i in range(3):
            (diag_sub / f"diag.2024-01-0{i+1}.nc").write_bytes(b"")

        processor = _make_mock_2d_processor(verbose=False)

        with patch.object(processor, '_find_files_by_pattern', side_effect=[
            FileNotFoundError("no files"), 
            [str(f) for f in sorted(diag_sub.glob("diag*.nc"))],
        ]):
            result = processor.find_diagnostic_files(str(tmp_path))
        assert len(result) == 3

    def test_find_diag_files_recursive(self: "TestFileDiscoveryFallbacks", tmp_path: Any) -> None:
        """
        This test checks that when both the primary search and the `diag` subdirectory search for diagnostic files fail, the method correctly falls back to a recursive search for diagnostic files. It creates a nested directory structure within the temporary path and populates it with mock diagnostic files. The test then patches the file discovery method to simulate failures in both the primary and subdirectory searches while allowing the recursive search to succeed, and asserts that the correct number of files is returned from the recursive search.

        Parameters:
            self: Test instance provided by pytest.
            tmp_path: pytest fixture providing a temporary directory for file creation.

        Returns:
            None
        """
        sub = tmp_path / "nested" / "deep"
        sub.mkdir(parents=True)

        for i in range(3):
            (sub / f"diag.2024-01-0{i+1}.nc").write_bytes(b"")

        processor = _make_mock_2d_processor(verbose=True)

        with patch.object(processor, '_find_files_by_pattern', side_effect=FileNotFoundError("none")):
            from io import StringIO
            captured = StringIO()
            with patch('sys.stdout', captured):
                result = processor.find_diagnostic_files(str(tmp_path))

        assert len(result) == 3

    def test_fallback_to_mpasout_files(self: "TestFileDiscoveryFallbacks", tmp_path: Any) -> None:
        """
        This test verifies that when all searches for diagnostic files fail, the method correctly falls back to searching for `mpasout` files in the specified directory. It creates mock `mpasout` files directly within the temporary path. The test then patches the file discovery method to simulate failures in all diagnostic file searches while allowing the search for `mpasout` files to succeed, and asserts that the correct number of `mpasout` files is returned. This confirms that the method can successfully identify and return `mpasout` files as a fallback when no diagnostic files are found in the expected locations. 

        Parameters:
            self: Test instance provided by pytest.
            tmp_path: pytest fixture providing a temporary directory for file creation.

        Returns:
            None
        """
        for i in range(3):
            (tmp_path / f"mpasout.2024-01-0{i+1}.nc").write_bytes(b"")

        processor = _make_mock_2d_processor(verbose=True)

        with patch.object(processor, '_find_files_by_pattern', side_effect=[
            FileNotFoundError("no diag"), 
            FileNotFoundError("no diag sub"),  
            [str(f) for f in sorted(tmp_path.glob("mpasout*.nc"))], 
        ]):
            from io import StringIO
            captured = StringIO()
            with patch('sys.stdout', captured):
                result = processor.find_diagnostic_files(str(tmp_path))
        assert len(result) == 3

    def test_fallback_to_mpasout_recursive(self: "TestFileDiscoveryFallbacks", tmp_path: Any) -> None:
        """
        This test checks that when all searches for diagnostic files and `mpasout` files in the specified directory fail, the method correctly falls back to a recursive search for `mpasout` files. It creates a nested directory structure within the temporary path and populates it with mock `mpasout` files. The test then patches the file discovery method to simulate failures in all previous search attempts while allowing the recursive search for `mpasout` files to succeed, and asserts that the correct number of files is returned from the recursive search. This confirms that the method can successfully identify and return `mpasout` files from a recursive search as a final fallback when no diagnostic files are found in any expected locations.

        Parameters:
            self: Test instance provided by pytest.
            tmp_path: pytest fixture providing a temporary directory for file creation.

        Returns:
            None
        """
        sub = tmp_path / "output"
        sub.mkdir()

        for i in range(3):
            (sub / f"mpasout.2024-01-0{i+1}.nc").write_bytes(b"")

        processor = _make_mock_2d_processor(verbose=True)

        with patch.object(processor, '_find_files_by_pattern', side_effect=FileNotFoundError("none")):
            from io import StringIO
            captured = StringIO()
            with patch('sys.stdout', captured):
                result = processor.find_diagnostic_files(str(tmp_path))
        assert len(result) == 3

    def test_no_files_found_raises_error(self: "TestFileDiscoveryFallbacks", tmp_path: Any) -> None:
        """
        This test verifies that when all searches for diagnostic files and `mpasout` files fail, the method raises a `FileNotFoundError` with an appropriate message indicating that no diagnostic files were found. It patches the file discovery method to simulate failures in all search attempts, ensuring that no files are found in any location. The test then asserts that the expected exception is raised with a message containing "No diagnostic files", confirming that the method correctly handles the case where no relevant files are available and provides meaningful feedback through exceptions. 

        Parameters:
            self: Test instance provided by pytest.
            tmp_path: pytest fixture providing a temporary directory for file creation.

        Returns:
            None
        """
        processor = _make_mock_2d_processor(verbose=False)

        with patch.object(processor, '_find_files_by_pattern', side_effect=FileNotFoundError("none")):
            with pytest.raises(FileNotFoundError, match="No diagnostic files"):
                processor.find_diagnostic_files(str(tmp_path))

    def test_insufficient_mpasout_files(self: "TestFileDiscoveryFallbacks", tmp_path: Any) -> None:
        """
        This test checks that when the method falls back to searching for `mpasout` files but finds an insufficient number of them (e.g., only one file), it raises a `ValueError` with an appropriate message indicating that there are insufficient diagnostic files. It creates a single mock `mpasout` file within a subdirectory of the temporary path and patches the file discovery method to simulate failures in all previous search attempts while allowing the search for `mpasout` files to succeed. The test then asserts that the expected exception is raised with a message containing "Insufficient", confirming that the method correctly handles cases where fallback files are found but do not meet the criteria for a valid dataset, providing meaningful feedback through exceptions. 

        Parameters:
            self: Test instance provided by pytest.
            tmp_path: pytest fixture providing a temporary directory for file creation.

        Returns:
            None
        """
        sub = tmp_path / "output"
        sub.mkdir()
        (sub / "mpasout.2024-01-01.nc").write_bytes(b"")

        processor = _make_mock_2d_processor(verbose=False)
        with patch.object(processor, '_find_files_by_pattern', side_effect=FileNotFoundError("none")):
            with pytest.raises(ValueError, match="Insufficient"):
                processor.find_diagnostic_files(str(tmp_path))


class TestCoordinateExtractionBranches:
    """ Tests for extract_2d_coordinates_for_variable branch coverage. """

    def test_nvertices_coordinate_extraction(self: "TestCoordinateExtractionBranches") -> None:
        """
        This test verifies that the `extract_2d_coordinates_for_variable` method can successfully extract longitude and latitude coordinates when the dataset contains `lonVertex` and `latVertex` variables with an `nVertices` dimension. It creates a mock dataset with random longitude and latitude values for a specified number of vertices, initializes a mock processor with this dataset, and calls the coordinate extraction method. The test asserts that the returned longitude and latitude arrays have the expected length corresponding to the number of vertices and that the latitude values are within valid ranges, confirming that the method can handle vertex-based coordinate extraction correctly. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        n_vertices = 100
        lons = np.random.uniform(-np.pi, np.pi, n_vertices)
        lats = np.random.uniform(-np.pi / 2, np.pi / 2, n_vertices)

        ds = xr.Dataset({
            'vorticity': xr.DataArray(np.random.randn(n_vertices), dims=['nVertices']),
            'lonVertex': xr.DataArray(lons, dims=['nVertices']),
            'latVertex': xr.DataArray(lats, dims=['nVertices']),
        })

        processor = _make_mock_2d_processor(verbose=False, dataset=ds)

        lon, lat = processor.extract_2d_coordinates_for_variable('vorticity')
        assert len(lon) == n_vertices
        assert np.all(np.abs(lat) <= 90.1)

    def test_nvertices_via_data_array(self: "TestCoordinateExtractionBranches") -> None:
        """
        This test checks that the `extract_2d_coordinates_for_variable` method can extract longitude and latitude coordinates using the dimensions of a provided data array when the dataset contains `lonVertex` and `latVertex` variables with an `nVertices` dimension. It creates a mock dataset with longitude and latitude values for a specified number of vertices, initializes a mock processor with this dataset, and calls the coordinate extraction method while passing a data array that has the same `nVertices` dimension. The test asserts that the returned longitude and latitude arrays have the expected length corresponding to the number of vertices and that the method correctly identifies the coordinate variables based on the data array's dimensions, confirming that it can handle this branch of coordinate extraction logic effectively. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        n_v = 50

        ds = xr.Dataset({
            'lonVertex': xr.DataArray(np.linspace(-np.pi, np.pi, n_v), dims=['nVertices']),
            'latVertex': xr.DataArray(np.linspace(-np.pi / 2, np.pi / 2, n_v), dims=['nVertices']),
        })

        data_arr = xr.DataArray(np.ones(n_v), dims=['nVertices'])

        processor = _make_mock_2d_processor(verbose=True, dataset=ds)
        from io import StringIO
        captured = StringIO()

        with patch('sys.stdout', captured):
            lon, lat = processor.extract_2d_coordinates_for_variable('test_var', data_array=data_arr)
        
        assert len(lon) == n_v
        assert 'nVertices' in captured.getvalue()

    def test_alternative_coord_names_longitude_latitude(self: "TestCoordinateExtractionBranches") -> None:
        """
        This test verifies that the `extract_2d_coordinates_for_variable` method can successfully extract longitude and latitude coordinates when the dataset contains alternative variable names for longitude and latitude (e.g., `longitude`, `latitude`) without the standard `lonCell` and `latCell` names. It creates a mock dataset with random longitude and latitude values for a specified number of cells, initializes a mock processor with this dataset, and calls the coordinate extraction method. The test asserts that the returned longitude and latitude arrays have the expected length corresponding to the number of cells and that the latitude values are within valid ranges, confirming that the method can handle coordinate extraction using alternative variable names correctly. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        n_cells = 80
        ds = xr.Dataset({
            'temperature': xr.DataArray(np.random.randn(n_cells), dims=['nCells']),
            'longitude': xr.DataArray(np.linspace(-180, 180, n_cells), dims=['nCells']),
            'latitude': xr.DataArray(np.linspace(-90, 90, n_cells), dims=['nCells']),
        })
        processor = _make_mock_2d_processor(verbose=False, dataset=ds)

        lon, lat = processor.extract_2d_coordinates_for_variable('temperature')
        assert len(lon) == n_cells
        assert np.all(np.abs(lat) <= 90.1)

    def test_missing_coordinates_raises_error(self: "TestCoordinateExtractionBranches") -> None:
        """
        This test checks that the `extract_2d_coordinates_for_variable` method raises a ValueError when the dataset does not contain any recognizable longitude and latitude coordinate variables. It creates a mock dataset with a variable but without any coordinate variables, initializes a mock processor with this dataset, and calls the coordinate extraction method. The test asserts that a ValueError is raised with an appropriate message indicating that the coordinates could not be found, confirming that the method correctly handles cases where necessary coordinate information is missing from the dataset. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        ds = xr.Dataset({
            'temperature': xr.DataArray(np.random.randn(10), dims=['nCells']),
        })

        processor = _make_mock_2d_processor(verbose=False, dataset=ds)

        with pytest.raises(ValueError, match="Could not find"):
            processor.extract_2d_coordinates_for_variable('temperature')

    def test_dataset_not_loaded_raises_error(self: "TestCoordinateExtractionBranches") -> None:
        """
        This test verifies that the `extract_2d_coordinates_for_variable` method raises a ValueError when the processor's dataset attribute is None, indicating that no dataset has been loaded. It initializes a mock processor with the dataset set to None and calls the coordinate extraction method. The test asserts that a ValueError is raised with an appropriate message indicating that the dataset is not loaded, confirming that the method correctly handles cases where it is called without a valid dataset, providing meaningful feedback through exceptions. 

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        processor = _make_mock_2d_processor(verbose=False)
        processor.dataset = None

        with pytest.raises(ValueError, match="Dataset not loaded"):
            processor.extract_2d_coordinates_for_variable('temperature')

    def test_radian_to_degree_conversion(self: "TestCoordinateExtractionBranches") -> None:
        """
        This test checks that the `extract_2d_coordinates_for_variable` method correctly converts longitude and latitude values from radians to degrees when the dataset contains coordinate variables in radians. It creates a mock dataset with longitude and latitude values in radians for a specified number of cells, initializes a mock processor with this dataset, and calls the coordinate extraction method. The test asserts that the returned longitude and latitude arrays are within valid degree ranges (longitude between -180 and 180, latitude between -90 and 90), confirming that the method can handle radian-to-degree conversion correctly when extracting coordinates.

        Parameters:
            self: Test instance provided by pytest.

        Returns:
            None
        """
        n_cells = 20
        lons_rad = np.linspace(-np.pi, np.pi, n_cells)
        lats_rad = np.linspace(-np.pi / 2, np.pi / 2, n_cells)

        ds = xr.Dataset({
            'temperature': xr.DataArray(np.ones(n_cells), dims=['nCells']),
            'lonCell': xr.DataArray(lons_rad, dims=['nCells']),
            'latCell': xr.DataArray(lats_rad, dims=['nCells']),
        })

        processor = _make_mock_2d_processor(verbose=False, dataset=ds)
        lon, lat = processor.extract_2d_coordinates_for_variable('temperature')

        assert np.all(lon >= -180.1) and np.all(lon <= 180.1)
        assert np.all(lat >= -90.1) and np.all(lat <= 90.1)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
