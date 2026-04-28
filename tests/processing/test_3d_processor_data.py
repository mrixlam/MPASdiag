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
import numpy as np
from typing import Any
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


class TestLoad3DData:
    """ Test 3D data loading. """
    
    @classmethod
    def setup_class(cls: 'TestLoad3DData') -> None:
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
    
    def test_load_with_xarray(self: 'TestLoad3DData') -> None:
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
    
    def test_load_with_uxarray(self: 'TestLoad3DData') -> None:
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
    
    def test_load_verbose_mode(self: 'TestLoad3DData') -> None:
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
    def setup_class(cls: 'TestGetAvailable3DVariables') -> None:
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
    
    
    def test_verbose_output(self: 'TestGetAvailable3DVariables') -> None:
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
    

class TestGet3DVariableData:
    """ Test variable data extraction. """
    
    def setup_method(self: 'TestGet3DVariableData') -> None:
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
    
    def test_extract_real_3d_variable_at_level(self: 'TestGet3DVariableData') -> None:
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
    
    def test_extract_real_3d_variable_surface(self: 'TestGet3DVariableData') -> None:
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
    
    def test_extract_real_3d_variable_top(self: 'TestGet3DVariableData') -> None:
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
    
    
    def test_string_level_surface(self: 'TestGet3DVariableData') -> None:
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
    
    def test_string_level_top(self: 'TestGet3DVariableData') -> None:
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
    
    
    def test_pressure_level_above_surface(self: 'TestGet3DVariableData') -> None:
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
    
    def test_pressure_level_below_top(self: 'TestGet3DVariableData') -> None:
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
    
    def test_pressure_interpolation(self: 'TestGet3DVariableData') -> None:
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
    
    
    def test_verbose_output(self: 'TestGet3DVariableData') -> None:
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
    
    def _create_mock_dataset(self: 'TestGet3DVariableData') -> Any:
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
    
    def _create_mock_dataset_with_pressure(self: 'TestGet3DVariableData') -> Any:
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
        
        def pressure_base_isel_side_effect(*args: Any, 
                                           **kwargs: Any) -> Any:
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
        
        def pressure_p_isel_side_effect(*args: Any, 
                                        **kwargs: Any) -> Any:
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
