#!/usr/bin/env python3

"""
MPASdiag Test Suite: Tests for 3D Atmospheric Data Processing in MPASdiag

This module contains a comprehensive set of unit tests for the MPAS3DProcessor class, which is responsible for loading, processing, and extracting 3D atmospheric data from MPAS model output. The tests cover a range of functionalities including coordinate extraction, data loading with different backends, variable discovery, pressure level interpolation, and attribute handling. Both edge cases and typical usage scenarios are tested to ensure robustness and correctness of the processor's behavior when working with real MPAS datasets.

Author: Rubaiat Islam
Institution: Mesoscale & Microscale Meteorology Laboratory, NCAR
Email: mrislam@ucar.edu
Date: February 2026
Version: 1.0.0
"""
# Load standard libraries
import os
import sys
import pytest
import tempfile
import numpy as np
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock, patch

from mpasdiag.processing.processors_3d import MPAS3DProcessor
from tests.test_data_helpers import get_mpas_data_paths, check_mpas_data_available, load_mpas_3d_processor, assert_expected_public_methods

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')


def make_getitem(mapping: dict, 
                 default: Any = None) -> Any:
    """
    This function returns a __getitem__ side_effect function for a given mapping. It allows for flexible mocking of xarray Datasets or DataArrays by returning values based on the provided mapping dictionary. If a key is not found in the mapping, it returns a default value (which can be a MagicMock if not specified), enabling tests to simulate various dataset configurations and variable access patterns.
    
    Parameters:
        mapping (dict): A dictionary mapping keys to values.
        default (Any): The default value to return if the key is not found.

    Returns:
        Any: A function suitable for use as a __getitem__ side_effect.
    """
    def _getitem(key: str) -> Any:
        return mapping.get(key, MagicMock() if default is None else default)
    return _getitem


def make_contains(keys: Any) -> Any:
    """
    This function returns a __contains__ side_effect function for a given collection of keys. It allows for flexible mocking of the __contains__ method in xarray Datasets or DataArrays by checking if a key is present in the provided collection. This is useful for simulating the presence or absence of variables or coordinates in a dataset during testing. 

    Parameters:
        keys (Any): A collection of keys to check against.

    Returns:
        Any: A function suitable for use as a __contains__ side_effect.
    """
    key_set = set(keys)
    def _contains(key: str) -> bool:
        """ Check if the key is in the provided collection of keys. """
        return key in key_set
    return _contains


def make_grid_getitem(lon_values: Any, 
                      lat_values: Any) -> Any:
    """
    This function returns a __getitem__ side_effect function that provides longitude and latitude values based on the key. It checks if the key contains 'lon' or 'lat' (case-insensitive) and returns a mock coordinate object with the corresponding values. This is particularly useful for testing the coordinate extraction logic in MPAS3DProcessor when working with grid datasets. 

    Parameters:
        lon_values (Any): The longitude values to return for keys containing 'lon'.
        lat_values (Any): The latitude values to return for keys containing 'lat'.

    Returns:
        Any: A function suitable for use as a __getitem__ side_effect.
    """
    def _getitem(key: str) -> Any:
        """ Return longitude or latitude values based on the key. """
        mock_coord = MagicMock()
        mock_coord.values = lon_values if 'lon' in key.lower() else lat_values
        return mock_coord
    return _getitem


def make_getitem_with_raise(raise_key: str, 
                            exc: Exception, 
                            default: Any = None) -> Any:
    """
    This function returns a __getitem__ side_effect function that raises a specified exception when a particular key is accessed. For all other keys, it returns a default value (which can be a MagicMock if not specified). This is useful for testing the error handling logic in MPAS3DProcessor when certain expected variables or coordinates are missing or inaccessible in the dataset. 

    Parameters:
        raise_key (str): The key for which the exception should be raised.
        exc (Exception): The exception to raise when the raise_key is accessed.
        default (Any): The default value to return if the key is not the raise_key.

    Returns:
        Any: A function suitable for use as a __getitem__ side_effect.
    """
    def _getitem(key: str) -> Any:
        """ Raise the specified exception if the key matches raise_key, otherwise return a default value. """
        if key == raise_key:
            raise exc
        return MagicMock() if default is None else default
    return _getitem


class TestFindMpasoutFiles:
    """ Test MPAS output file discovery. """
    
    @classmethod
    def setup_class(cls) -> None:
        """
        This class-level setup method checks for the availability of MPAS test data and retrieves the necessary paths for testing. If the test data is not found, it skips all tests in this class. By storing the paths in a class variable, it allows individual test methods to access the required file paths without needing to perform redundant checks or retrievals, ensuring efficient use of resources during testing. This setup is crucial for tests that depend on specific MPAS output files being present in the test data directory. 

        Parameters:
            None

        Returns:
            None
        """
        if not check_mpas_data_available():
            pytest.skip("MPAS test data not found")
            return
        
        cls.paths = get_mpas_data_paths()
    
    def setup_method(self: "TestFindMpasoutFiles") -> None:
        """
        This setup method initializes an `MPAS3DProcessor` instance using the grid file path from the test data. By creating a processor instance before each test method runs, it ensures that the tests have access to a properly initialized processor that can be used to call the `find_mpasout_files` method. This setup allows the subsequent test methods to focus on testing the file discovery functionality without worrying about processor initialization issues. If the test data is not available, the tests will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        self.processor = MPAS3DProcessor(str(self.paths['grid_file']), verbose=False)
        assert_expected_public_methods(self.processor, 'MPAS3DProcessor')
    
    def test_find_in_mpasout_subdirectory(self: "TestFindMpasoutFiles") -> None:
        """
        This test verifies that the `find_mpasout_files` method can successfully discover MPAS output files located within a `mpasout` subdirectory of the specified path. By invoking the method with the path to the `mpasout` directory in the test data, the test asserts that at least two files are found, that the files are sorted, and that their names contain "mpasout", confirming that the file discovery logic can correctly identify and retrieve MPAS output files organized within a specific subdirectory structure. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        files = self.processor.find_mpasout_files(str(self.paths['mpasout_dir']))

        assert len(files) >= 2
        assert files == sorted(files)

        for f in files:
            assert 'mpasout' in f.lower()
    
    def test_find_files_recursive(self: "TestFindMpasoutFiles") -> None:
        """
        This test checks that the `find_mpasout_files` method can discover MPAS output files that are located in nested subdirectories within the specified path. By creating a temporary directory structure with a `mpasout` subdirectory containing MPAS output files, and invoking the method with the path to the main directory, the test asserts that at least two files are found, confirming that the file discovery logic can handle recursive searching through subdirectories to locate MPAS output files. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        self.processor.verbose = True
        files = self.processor.find_mpasout_files(str(self.paths['mpasout_dir']))
        assert len(files) >= 2
    
    def test_find_in_main_directory(self: "TestFindMpasoutFiles") -> None:
        """
        This test verifies that the `find_mpasout_files` method can successfully discover MPAS output files that are located directly in the specified main directory, without being nested in a `mpasout` subdirectory. By invoking the method with the path to the main directory in the test data, the test asserts that at least two files are found, confirming that the file discovery logic can correctly identify and retrieve MPAS output files organized directly within the main directory structure. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        files = self.processor.find_mpasout_files(str(self.paths['mpasout_dir']))
        assert len(files) >= 2
    
    def test_insufficient_files_error(self: "TestFindMpasoutFiles") -> None:
        """
        This test checks that the `find_mpasout_files` method raises a `ValueError` when it finds fewer than two MPAS output files in the specified directory. By creating a temporary directory with only one MPAS output file and invoking the method, the test ensures that it correctly identifies the issue of insufficient files and raises an appropriate error message containing "Insufficient", confirming that the method can handle cases where not enough relevant files are present to proceed with processing. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            nested = Path(temp_dir) / 'data'
            nested.mkdir()
            (nested / 'mpasout.2024-01-01.nc').touch()
            
            with pytest.raises(ValueError) as exc_info:
                self.processor.find_mpasout_files(temp_dir)
            assert 'Insufficient' in str(exc_info.value)
    
    def test_no_files_error(self: "TestFindMpasoutFiles") -> None:
        """
        This test verifies that the `find_mpasout_files` method raises a `FileNotFoundError` when it does not find any MPAS output files in the specified directory. By creating an empty temporary directory and invoking the method, the test ensures that it correctly identifies the absence of relevant files and raises an appropriate error message containing "No MPAS output files", confirming that the method can handle cases where no files are present to proceed with processing. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None

        Returns:
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError) as exc_info:
                self.processor.find_mpasout_files(temp_dir)
            assert 'No MPAS output files' in str(exc_info.value)


class TestLoad3DData:
    """ Test 3D data loading. """
    
    @classmethod
    def setup_class(cls) -> None:
        """
        This class-level setup method checks for the availability of MPAS test data and retrieves the necessary paths for testing. If the test data is not found, it skips all tests in this class. By storing the paths in a class variable, it allows individual test methods to access the required file paths without needing to perform redundant checks or retrievals, ensuring efficient use of resources during testing. This setup is crucial for tests that depend on loading actual MPAS output data to verify the functionality of the `load_3d_data` method. 

        Parameters:
            None

        Returns:
            None
        """
        if not check_mpas_data_available():
            pytest.skip("MPAS test data not found")
            return
        
        cls.paths = get_mpas_data_paths()
    
    def test_load_with_xarray(self: "TestLoad3DData") -> None:
        """
        This test verifies that the `load_3d_data` method can successfully load MPAS output data using the `xarray` backend when `use_pure_xarray=True`. By invoking the method with the path to the MPAS output directory and asserting that the processor's `data_type` is set to 'xarray' and that the dataset contains data variables, the test confirms that the loading process works correctly with the xarray backend. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS3DProcessor(str(self.paths['grid_file']), verbose=False)
        assert_expected_public_methods(processor, 'MPAS3DProcessor')

        result = processor.load_3d_data(str(self.paths['mpasout_dir']), use_pure_xarray=True)
        
        assert result == processor
        assert processor.data_type == 'xarray'
        assert processor.dataset is not None
        assert len(processor.dataset.data_vars) > 0
    
    def test_load_with_uxarray(self: "TestLoad3DData") -> None:
        """
        This test checks that the `load_3d_data` method can successfully load MPAS output data using the `uxarray` backend when `use_pure_xarray=False`. By invoking the method with the path to the MPAS output directory and asserting that the processor's `data_type` is set to 'uxarray' or 'xarray' (depending on availability) and that the dataset contains data variables, the test confirms that the loading process works correctly with the uxarray backend. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS3DProcessor(str(self.paths['grid_file']), verbose=False)
        assert_expected_public_methods(processor, 'MPAS3DProcessor')

        result = processor.load_3d_data(str(self.paths['mpasout_dir']), use_pure_xarray=False)
        
        assert result == processor
        assert processor.data_type in ['uxarray', 'xarray']
        assert processor.dataset is not None
        assert len(processor.dataset.data_vars) > 0
    
    def test_load_verbose_mode(self: "TestLoad3DData") -> None:
        """
        This test verifies that when the `load_3d_data` method is called with `verbose=True`, it provides informative output about the loading process. By invoking the method in verbose mode and capturing the standard output, the test checks that messages related to loading data (e.g., "Loading 3D data", "using xarray", "using uxarray") are present, confirming that users receive feedback about the data loading process. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS3DProcessor(str(self.paths['grid_file']), verbose=True)
        assert_expected_public_methods(processor, 'MPAS3DProcessor')

        result = processor.load_3d_data(str(self.paths['mpasout_dir']), use_pure_xarray=True)

        assert result == processor
        assert processor.dataset is not None


class TestGetAvailable3DVariables:
    """ Test 3D variable detection. """
    
    @classmethod
    def setup_class(cls) -> None:
        """
        This class-level setup method checks for the availability of MPAS test data, retrieves the necessary paths, and initializes an `MPAS3DProcessor` instance with the loaded dataset. If the test data is not found, it skips all tests in this class. By storing the processor instance in a class variable, it allows individual test methods to access the initialized processor without needing to perform redundant checks or initializations, ensuring efficient use of resources during testing. This setup is crucial for tests that depend on having a loaded MPAS dataset to verify the functionality of the `get_available_3d_variables` method. 

        Parameters:
            None

        Returns:
            None
        """
        if not check_mpas_data_available():
            pytest.skip("MPAS test data not found")
            return
        
        cls.paths = get_mpas_data_paths()
        cls.processor = load_mpas_3d_processor(verbose=False)
    
    def test_find_3d_variables(self: "TestGetAvailable3DVariables") -> None:
        """
        This test verifies that the `get_available_3d_variables` method can successfully identify and return a list of available 3D variables from the loaded MPAS dataset. By invoking the method and asserting that the returned list contains expected 3D variable names (e.g., 'theta', 'w', 'rho', 'pressure', 'uReconstructZonal', 'uReconstructMeridional'), the test confirms that the variable detection logic is functioning correctly and can recognize common 3D variables present in MPAS output data. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        vars_list = self.processor.get_available_3d_variables()
        
        assert len(vars_list) > 0

        has_3d_vars = any(v in ['theta', 'w', 'rho', 'pressure', 'uReconstructZonal', 'uReconstructMeridional'] 
                          for v in vars_list)
        
        assert has_3d_vars, f"Expected 3D variables not found. Found: {vars_list}"
    
    def test_verbose_output(self: "TestGetAvailable3DVariables") -> None:
        """
        This test checks that when the `get_available_3d_variables` method is called in verbose mode, it provides informative output about the variable detection process. By invoking the method with `verbose=True` and capturing the standard output, the test checks that messages related to finding 3D variables (e.g., "Finding available 3D variables", "available 3D variables") are present, confirming that users receive feedback about the variable detection process. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS3DProcessor(str(self.paths['grid_file']), verbose=True)
        assert_expected_public_methods(processor, 'MPAS3DProcessor')
        processor.load_3d_data(str(self.paths['mpasout_dir']))
        
        vars_list = processor.get_available_3d_variables()
        assert len(vars_list) > 0
    
    def test_without_dataset_raises_error(self: "TestGetAvailable3DVariables") -> None:
        """
        This test verifies that if the `dataset` attribute of the processor is not set (i.e., remains `None`), the `get_available_3d_variables` method raises a `ValueError` indicating that the dataset is not loaded. By explicitly setting `processor.dataset` to `None` and invoking the method, the test ensures that it correctly identifies the missing dataset and raises an appropriate error message. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS3DProcessor(str(self.paths['grid_file']), verbose=False)
        assert_expected_public_methods(processor, 'MPAS3DProcessor')

        processor.dataset = None

        with pytest.raises(ValueError) as exc_info:
            processor.get_available_3d_variables()

        assert 'not loaded' in str(exc_info.value)


class TestGet3DVariableData:
    """ Test variable data extraction. """
    
    def setup_method(self: "TestGet3DVariableData") -> None:
        """
        This method-level setup initializes an `MPAS3DProcessor` instance with a mocked grid file path and `verbose=False`. The filesystem existence check is patched to always return True, allowing the test to focus on the processor initialization logic without relying on actual files. This setup provides a consistent starting point for the subsequent tests that will verify the behavior of 3D variable data extraction methods. If the test data is not available, the tests will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        with patch('os.path.exists', return_value=True):
            self.processor = MPAS3DProcessor('test_grid.nc', verbose=False)
            assert_expected_public_methods(self.processor, 'MPAS3DProcessor')
    
    def test_extract_real_3d_variable_at_level(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that the `get_3d_variable_data` method can successfully extract a 3D variable at a specific model level from the real MPAS dataset. By loading the dataset, retrieving an available 3D variable, and invoking the method with a valid level index, the test asserts that the returned data array is present and contains values, confirming that the method can correctly access and slice 3D variable data based on a specified vertical level. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = load_mpas_3d_processor(verbose=False)
        
        vars_3d = processor.get_available_3d_variables()
        assert len(vars_3d) > 0, "No 3D variables found"
        
        var_name = vars_3d[0]  
        data = processor.get_3d_variable_data(var_name, level=10, time_index=0)
        
        assert data is not None
        assert hasattr(data, 'values')
        assert len(data.values) > 0
    
    def test_extract_real_3d_variable_surface(self: "TestGet3DVariableData") -> None:
        """
        This test checks that the `get_3d_variable_data` method can successfully extract a 3D variable at the surface model level using the special string level 'surface' from the real MPAS dataset. By loading the dataset, retrieving an available 3D variable, and invoking the method with `level='surface'`, the test asserts that the returned data array is present and contains values, confirming that the method can correctly access and slice 3D variable data based on the surface level. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = load_mpas_3d_processor(verbose=False)
        vars_3d = processor.get_available_3d_variables()
        var_name = vars_3d[0]
        
        data = processor.get_3d_variable_data(var_name, level='surface', time_index=0)
        assert data is not None
        assert len(data.values) > 0
    
    def test_extract_real_3d_variable_top(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that the `get_3d_variable_data` method can successfully extract a 3D variable at the top model level using the special string level 'top' from the real MPAS dataset. By loading the dataset, retrieving an available 3D variable, and invoking the method with `level='top'`, the test asserts that the returned data array is present and contains values, confirming that the method can correctly access and slice 3D variable data based on the top level. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = load_mpas_3d_processor(verbose=False)
        vars_3d = processor.get_available_3d_variables()
        var_name = vars_3d[0]
        
        data = processor.get_3d_variable_data(var_name, level='top', time_index=0)
        assert data is not None
        assert len(data.values) > 0
    
    def test_without_dataset_raises_error(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that if the `dataset` attribute of the processor is not set (i.e., remains `None`), the `get_3d_variable_data` method raises a `ValueError` indicating that the dataset is not loaded. By explicitly setting `processor.dataset` to `None` and invoking the method with a variable name and level, the test ensures that it correctly identifies the missing dataset and raises an appropriate error message. If the test data is not available, the test will be skipped to avoid false failures.    

        Parameters:
            None

        Returns:
            None
        """
        self.processor.dataset = None
        with pytest.raises(ValueError):
            self.processor.get_3d_variable_data('theta', 0)
    
    def test_variable_not_found(self: "TestGet3DVariableData") -> None:
        """
        This test checks that if the requested variable name is not found in the dataset's data variables, the `get_3d_variable_data` method raises a `ValueError` indicating that the variable is not found. By mocking the dataset to contain a different variable and invoking the method with a variable name that does not exist in the dataset, the test ensures that it correctly identifies the missing variable and raises an appropriate error message. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'pressure': True}
        self.processor.dataset = mock_ds
        
        with pytest.raises(ValueError) as exc_info:
            self.processor.get_3d_variable_data('theta', 0)

        assert 'not found' in str(exc_info.value)
    
    def test_not_3d_variable(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that if the requested variable does not have the expected 3D dimensions (e.g., it has dimensions like `nCells` and `Time` instead of `nCells`, `nVertLevels`, and `Time`), the `get_3d_variable_data` method raises a `ValueError` indicating that the variable is not a 3D variable. By mocking the dataset to contain a variable with incorrect dimensions and invoking the method, the test ensures that it correctly identifies the issue with the variable's dimensionality and raises an appropriate error message. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'t2m': True}
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'Time': 10}
        mock_ds.__getitem__.return_value = mock_var
        self.processor.dataset = mock_ds
        
        with pytest.raises(ValueError) as exc_info:
            self.processor.get_3d_variable_data('t2m', 0)
        assert 'not a 3D' in str(exc_info.value)
    
    def test_integer_level(self: "TestGet3DVariableData") -> None:
        """
        This test checks that the `get_3d_variable_data` method can successfully extract a 3D variable at a specific integer model level from a mocked dataset. By creating a mock dataset with a variable that has the expected 3D dimensions and invoking the method with a valid integer level index, the test asserts that the returned data is present, confirming that the method can correctly access and slice 3D variable data based on a specified vertical level. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset()
        self.processor.dataset = mock_ds
        self.processor.data_type = 'xarray'
        
        data = self.processor.get_3d_variable_data('theta', level=10, time_index=0)
        assert data is not None
    
    def test_level_exceeds_maximum(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that if an integer level index is requested that exceeds the maximum available vertical levels in the dataset, the `get_3d_variable_data` method raises a `ValueError` indicating that the requested level exceeds available levels. By creating a mock dataset with a specific number of vertical levels and invoking the method with a level index that is out of bounds, the test ensures that it correctly identifies the issue with the requested level and raises an appropriate error message. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset()
        self.processor.dataset = mock_ds
        
        with pytest.raises(ValueError) as exc_info:
            self.processor.get_3d_variable_data('theta', level=100)

        assert 'exceeds available levels' in str(exc_info.value)
    
    def test_string_level_surface(self: "TestGet3DVariableData") -> None:
        """
        This test checks that the `get_3d_variable_data` method can successfully extract a 3D variable at the surface model level using the special string level 'surface' from a mocked dataset. By creating a mock dataset with a variable that has the expected 3D dimensions and invoking the method with `level='surface'`, the test asserts that the returned data is present, confirming that the method can correctly access and slice 3D variable data based on the surface level. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset()
        self.processor.dataset = mock_ds
        self.processor.data_type = 'xarray'
        
        data = self.processor.get_3d_variable_data('theta', level='surface')
        assert data is not None
    
    def test_string_level_top(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that the `get_3d_variable_data` method can successfully extract a 3D variable at the top model level using the special string level 'top' from a mocked dataset. By creating a mock dataset with a variable that has the expected 3D dimensions and invoking the method with `level='top'`, the test asserts that the returned data is present, confirming that the method can correctly access and slice 3D variable data based on the top level. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset()
        self.processor.dataset = mock_ds
        self.processor.data_type = 'xarray'
        
        data = self.processor.get_3d_variable_data('theta', level='top')
        assert data is not None
    
    def test_invalid_string_level(self: "TestGet3DVariableData") -> None:
        """
        This test checks that if an invalid string is passed as the `level` argument (i.e., a string that is not 'surface' or 'top'), the `get_3d_variable_data` method raises a `ValueError` indicating that the level is unknown. By mocking a dataset and invoking the method with an unsupported string level, the test confirms that the method correctly identifies the invalid input and raises an appropriate error message. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset()
        self.processor.dataset = mock_ds
        
        with pytest.raises(ValueError) as exc_info:
            self.processor.get_3d_variable_data('theta', level='middle')
            
        assert 'Unknown level' in str(exc_info.value)
    
    def test_invalid_level_type(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that if an invalid type (e.g., a list) is passed as the `level` argument, the `get_3d_variable_data` method raises a `ValueError` indicating that the level is invalid. By mocking a dataset and invoking the method with an unsupported type for the level argument, the test confirms that the method correctly identifies the invalid input and raises an appropriate error message. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset()
        self.processor.dataset = mock_ds

        invalid_lev: Any = [1, 2]

        with pytest.raises(ValueError) as exc_info:
            self.processor.get_3d_variable_data('theta', level=invalid_lev)

        assert 'Invalid level' in str(exc_info.value)
    
    def test_pressure_level_no_data(self: "TestGet3DVariableData") -> None:
        """
        This test checks that if a pressure level is requested for interpolation but the necessary pressure data components are not available in the dataset, the `get_3d_variable_data` method raises a `ValueError` indicating that pressure data is not available. By mocking a dataset that lacks the required pressure variables and invoking the method with a pressure level, the test ensures that it correctly identifies the missing pressure data and raises an appropriate error message. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10}
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        mock_ds.__getitem__.return_value = mock_var
        mock_ds.__contains__.return_value = False
        self.processor.dataset = mock_ds
        
        with pytest.raises(ValueError) as exc_info:
            self.processor.get_3d_variable_data('theta', level=85000.0)

        assert 'pressure data not available' in str(exc_info.value)
    
    def test_pressure_level_above_surface(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that if a pressure level is requested that is above the surface level (i.e., a pressure value that is higher than the surface pressure), the `get_3d_variable_data` method raises a `ValueError` indicating that the requested pressure level is above the surface. By mocking a dataset with synthetic pressure data and invoking the method with a pressure level that exceeds the surface pressure, the test ensures that it correctly identifies the issue with the requested pressure level and raises an appropriate error message. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return

        processor = load_mpas_3d_processor(verbose=False)        
        data = processor.get_3d_variable_data('theta', level=0, time_index=0)

        assert data is not None
        assert hasattr(data, 'values')
        assert data.values.size > 0
        assert data.attrs['level_index'] == pytest.approx(0, abs=1e-3)
    
    def test_pressure_level_below_top(self: "TestGet3DVariableData") -> None:
        """
        This test checks that if a pressure level is requested that is below the top level (i.e., a pressure value that is lower than the pressure at the top level), the `get_3d_variable_data` method can successfully perform interpolation and return valid data. By loading real MPAS data, adding synthetic pressure components to the dataset, and invoking the method with a specific pressure level that is below the top level, the test asserts that the returned data array is present and contains values, confirming that the method can correctly access and interpolate 3D variable data based on pressure levels. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        processor = load_mpas_3d_processor(verbose=False)        
        nlevels = processor.dataset.sizes['nVertLevels']
        top_level = nlevels - 1        
        data = processor.get_3d_variable_data('theta', level=top_level, time_index=0)

        assert data is not None
        assert hasattr(data, 'values')
        assert data.values.size > 0
        assert data.attrs['level_index'] == top_level
        assert top_level > 50 
    
    def test_pressure_interpolation(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that the `get_3d_variable_data` method can successfully perform pressure interpolation when requested with a pressure level. By loading real MPAS data, adding synthetic pressure components to the dataset, and invoking the method with a specific pressure level (e.g., 50000 Pa), the test asserts that the returned data array is present and contains values, confirming that the method can correctly access and interpolate 3D variable data based on pressure levels. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return

        processor = load_mpas_3d_processor(verbose=False)
        
        if 'pressure' in processor.dataset:
            pressure = processor.dataset['pressure']
            processor.dataset['pressure_base'] = pressure * 0.8
            processor.dataset['pressure_p'] = pressure * 0.2
            
            data = processor.get_3d_variable_data('theta', level=50000.0, time_index=0)

            assert data is not None
            assert hasattr(data, 'values')
            assert data.values.size > 0
    
    def test_uxarray_data_type(self: "TestGet3DVariableData") -> None:
        """
        This test checks that the `get_3d_variable_data` method can successfully extract a 3D variable at a specific level when the processor's `data_type` is set to 'uxarray'. By creating a mock dataset with the expected structure and setting the processor's dataset and data type accordingly, the test invokes the method to extract a 3D variable at a specific level and asserts that the returned data is present, confirming that the method can correctly access 3D variable data when using the uxarray backend. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset()
        self.processor.dataset = mock_ds
        self.processor.data_type = 'uxarray'
        
        data = self.processor.get_3d_variable_data('theta', level=5)
        assert data is not None
    
    def test_verbose_output(self: "TestGet3DVariableData") -> None:
        """
        This test verifies that when the `get_3d_variable_data` method is called in verbose mode, it provides informative output about the data extraction process. By creating a mock dataset, setting the processor's dataset and data type, and invoking the method with `verbose=True`, the test checks that messages related to extracting 3D variable data (e.g., "Extracting 3D variable data", "at level", "interpolating to pressure level") are present in the captured standard output, confirming that users receive feedback about the data extraction process. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset()
        self.processor.dataset = mock_ds
        self.processor.data_type = 'xarray'
        self.processor.verbose = True
        
        data = self.processor.get_3d_variable_data('theta', level=10)
        assert data is not None
    
    def _create_mock_dataset(self: "TestGet3DVariableData") -> Any:
        """
        This helper method creates a mock dataset that includes a 3D variable with the expected dimensions and attributes for testing the `get_3d_variable_data` method. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with appropriate sizes and attributes. By setting up the necessary return values for indexing and slicing operations, this mock dataset allows tests to focus on the logic of extracting 3D variable data without relying on actual MPAS data files, enabling controlled testing of various scenarios related to variable extraction. 

        Parameters:
            None

        Returns:
            Any: A MagicMock object emulating an xarray Dataset suitable for tests.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10, 'nCells': 40962}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        mock_var.attrs = {'units': 'K'}
        
        mock_data = MagicMock()
        mock_data.values = np.random.randn(40962) + 300
        mock_data.attrs = {'units': 'K'}
        mock_data.ndim = 1
        mock_data.compute.return_value = mock_data
        
        mock_var.isel.return_value = mock_data
        mock_var.__getitem__.return_value = mock_data
        mock_ds.__getitem__.return_value = mock_var
        mock_ds.__contains__.return_value = True
        
        return mock_ds
    
    def _create_mock_dataset_with_pressure(self: "TestGet3DVariableData") -> Any:
        """
        This helper method creates a mock dataset that includes the necessary pressure components (`pressure_p` and `pressure_base`) for testing pressure interpolation in the `get_3d_variable_data` method. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with synthetic pressure data that allows for testing interpolation logic. By setting up the necessary return values for indexing, slicing, and arithmetic operations, this mock dataset enables controlled testing of scenarios related to pressure level extraction without relying on actual MPAS data files, ensuring that the interpolation logic can be thoroughly verified. 

        Parameters:
            None

        Returns:
            Any: A MagicMock object emulating an xarray Dataset with pressure components.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True, 'pressure_p': True, 'pressure_base': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10, 'nCells': 40962}
        
        pressure_vals = np.linspace(100000, 1000, 55)
        
        mock_total = MagicMock()
        mock_mean_result = MagicMock()
        mock_mean_result.values = pressure_vals  
        mock_total.mean.return_value = mock_mean_result
        
        mock_pressure_base_time = MagicMock()
        mock_pressure_p_time = MagicMock()
        
        mock_pressure_base_time.__add__ = lambda self, other: mock_total
        mock_pressure_base_time.__radd__ = lambda self, other: mock_total
        mock_pressure_p_time.__add__ = lambda self, other: mock_total
        mock_pressure_p_time.__radd__ = lambda self, other: mock_total
        
        mock_single_level_p = MagicMock()
        mock_single_level_base = MagicMock()
        mock_single_level_total = MagicMock()
        mock_single_mean = MagicMock()
        mock_single_mean.values = np.float64(95000.0)  
        mock_single_level_total.mean.return_value = mock_single_mean
        
        mock_single_level_base.__add__ = lambda self, other: mock_single_level_total
        mock_single_level_base.__radd__ = lambda self, other: mock_single_level_total
        mock_single_level_p.__add__ = lambda self, other: mock_single_level_total
        mock_single_level_p.__radd__ = lambda self, other: mock_single_level_total
        
        def pressure_base_isel_side_effect(*args: Any, **kwargs: Any) -> Any:
            """
            This side effect for `isel` on the mocked `pressure_base` object determines whether to return a single-level extract or the time-sliced object based on the presence of the 'nVertLevels' key in the keyword arguments. If 'nVertLevels' is present, it returns a MagicMock representing a single-level extract; otherwise, it returns a MagicMock representing the time-sliced pressure base data. This allows tests to simulate both scenarios of accessing pressure data at specific levels and across time without relying on actual MPAS data files. 

            Parameters:
                *args: Positional arguments passed to `isel`.
                **kwargs: Keyword arguments passed to `isel`.

            Returns:
                Any: A MagicMock either for a single-level extract or the time-sliced object.
            """
            if 'nVertLevels' in kwargs:
                return mock_single_level_base
            return mock_pressure_base_time
        
        def pressure_p_isel_side_effect(*args: Any, **kwargs: Any) -> Any:
            """
            This side effect for `isel` on the mocked `pressure_p` object determines whether to return a single-level extract or the time-sliced object based on the presence of the 'nVertLevels' key in the keyword arguments. If 'nVertLevels' is present, it returns a MagicMock representing a single-level extract; otherwise, it returns a MagicMock representing the time-sliced pressure p data. This allows tests to simulate both scenarios of accessing pressure data at specific levels and across time without relying on actual MPAS data files. 

            Parameters:
                *args: Positional arguments passed to `isel`.
                **kwargs: Keyword arguments passed to `isel`.

            Returns:
                Any: A MagicMock either for a single-level extract or the time-sliced object.
            """
            if 'nVertLevels' in kwargs:
                return mock_single_level_p
            return mock_pressure_p_time
        
        mock_pressure_base = MagicMock()
        mock_pressure_base.isel.side_effect = pressure_base_isel_side_effect
        
        mock_pressure_p = MagicMock()
        mock_pressure_p.isel.side_effect = pressure_p_isel_side_effect
        
        mock_theta = MagicMock()
        mock_theta.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        mock_theta.dims = ('Time', 'nCells', 'nVertLevels')
        
        theta_data = np.random.randn(40962, 55)
        theta_isel_result = MagicMock()
        theta_isel_result.values = theta_data
        theta_isel_result.attrs = {'units': 'K'}
        theta_isel_result.compute = lambda: theta_isel_result
        mock_theta.isel.return_value = theta_isel_result
        
        mock_ds.__getitem__.side_effect = make_getitem({'pressure_base': mock_pressure_base, 'pressure_p': mock_pressure_p, 'theta': mock_theta})
        mock_ds.__contains__.side_effect = make_contains(['theta', 'pressure_p', 'pressure_base'])
        
        return mock_ds


