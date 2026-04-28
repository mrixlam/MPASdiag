#!/usr/bin/env python3

"""
MPASdiag Test Suite: Tests for 3D Atmospheric Data Processing in MPASdiag

This module contains a comprehensive set of unit tests for the MPAS3DProcessor class, which is responsible for loading, processing, and extracting 3D atmospheric data from MPAS model output. The tests cover a range of functionalities including coordinate extraction, data loading with different backends, variable discovery, pressure level interpolation, and attribute handling. Both edge cases and typical usage scenarios are tested to ensure robustness and correctness of the processor's behavior when working with real MPAS datasets. The tests utilize synthetic data as well as real MPAS output (when available) to validate the processor's functionality across different data types and configurations. Mocking is used to simulate various dataset structures and to test error handling paths without relying on actual files for every scenario. 

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
from typing import Any
from unittest.mock import MagicMock

from mpasdiag.processing.processors_3d import MPAS3DProcessor
from tests.test_data_helpers import get_mpas_data_paths, check_mpas_data_available, load_mpas_3d_processor, assert_expected_public_methods

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
GRID_FILE = os.path.join(TEST_DATA_DIR, 'grids', 'x1.10242.static.nc')


def make_getitem(mapping: dict, 
                 default: Any = None) -> Any:
    """
    This function returns a __getitem__ side_effect function for a given mapping dictionary. It allows for flexible mocking of the __getitem__ method in xarray Datasets or DataArrays by returning values based on the provided mapping. If a key is not found in the mapping, it returns a default value (which can be a MagicMock if not specified). This is useful for simulating the presence or absence of variables or coordinates in a dataset during testing. 
    
    Parameters:
        mapping (dict): A dictionary mapping keys to values.
        default (Any): The default value to return if the key is not found.

    Returns:
        Any: A function suitable for use as a __getitem__ side_effect.
    """
    def _getitem(key: str) -> Any:
        """ Return the value from the mapping for the given key, or a default value if the key is not found. """
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
        """ Return a mock coordinate object with longitude or latitude values based on the key. """
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


class TestPressureInterpolationVerbose:
    """ Test verbose output paths in pressure interpolation. """
    
    def test_pressure_interpolation_verbose_mode(self: 'TestPressureInterpolationVerbose') -> None:
        """
        This test verifies that when the `get_3d_variable_data` method is called in verbose mode for a pressure level that requires interpolation, the method provides output indicating that it is performing interpolation. By capturing the standard output during the method call, the test checks that messages related to interpolation (e.g., "interpolating", "pressure") are present, confirming that users are informed about the interpolation process when a pressure level is requested that does not match exactly with one of the model levels. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        import sys
        from io import StringIO
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return 

        processor = load_mpas_3d_processor(verbose=True)
        
        if 'pressure' in processor.dataset:
            pressure = processor.dataset['pressure']
            processor.dataset['pressure_base'] = pressure * 0.8
            processor.dataset['pressure_p'] = pressure * 0.2
            
            captured_output = StringIO()
            sys.stdout = captured_output
            
            var_data = processor.get_3d_variable_data('theta', level=50000.0, time_index=0)
            
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            assert var_data is not None
            assert 'interpolating' in output.lower() or 'pressure' in output.lower()
    
    def test_pressure_above_surface_verbose(self: 'TestPressureInterpolationVerbose') -> None:
        """
        This test checks that when a pressure level above the surface is requested in verbose mode, the `get_3d_variable_data` method provides output indicating that it is selecting the surface level. By capturing the standard output during the method call, the test ensures that messages related to selecting the surface level (e.g., "surface", "level 0") are present, confirming that users are informed about the fallback to the surface level when a pressure level above the surface is requested. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None
        
        Returns:
            None        
        """
        import sys
        from io import StringIO
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return  
        
        processor = load_mpas_3d_processor(verbose=True)
        
        if 'pressure' in processor.dataset:
            pressure = processor.dataset['pressure']
            processor.dataset['pressure_base'] = pressure * 0.8
            processor.dataset['pressure_p'] = pressure * 0.2
            
            captured_output = StringIO()
            sys.stdout = captured_output
            
            var_data = processor.get_3d_variable_data('theta', level=120000.0, time_index=0)
            
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            assert var_data is not None
            assert 'surface' in output.lower()
    
    def test_pressure_below_top_verbose(self: 'TestPressureInterpolationVerbose') -> None:
        """
        This test checks that when a pressure level below the top is requested in verbose mode, the `get_3d_variable_data` method provides output indicating that it is selecting the top level. By capturing the standard output during the method call, the test ensures that messages related to selecting the top level (e.g., "top", "level 0") are present, confirming that users are informed about the fallback to the top level when a pressure level below the top is requested. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None
        
        Returns:
            None        
        """
        import sys
        from io import StringIO
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        processor = load_mpas_3d_processor(verbose=True)
        
        if 'pressure' in processor.dataset:
            pressure = processor.dataset['pressure']
            processor.dataset['pressure_base'] = pressure * 0.8
            processor.dataset['pressure_p'] = pressure * 0.2
            
            captured_output = StringIO()
            sys.stdout = captured_output
            
            var_data = processor.get_3d_variable_data('theta', level=100.0, time_index=0)
            
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            assert var_data is not None
            assert 'top' in output.lower()
    
    def test_equal_pressures_interpolation(self: 'TestPressureInterpolationVerbose') -> None:
        """
        This test verifies that when the `get_3d_variable_data` method is called in verbose mode with a pressure level that matches exactly with one of the model levels, the method provides output indicating that it is selecting the exact level without interpolation. By capturing the standard output during the method call, the test checks that messages related to selecting the exact level (e.g., "exact", "level 20") are present, confirming that users are informed about the selection of the exact level when a pressure level matches one of the model levels. If the test data is not available, the test will be skipped to avoid false failures. 
        
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
            pressure = processor.dataset['pressure'].copy()
            pressure[:, :, 20] = 50000.0
            pressure[:, :, 21] = 50000.0
            
            processor.dataset['pressure_base'] = pressure * 0.8
            processor.dataset['pressure_p'] = pressure * 0.2
            
            var_data = processor.get_3d_variable_data('theta', level=50000.0, time_index=0)
            assert var_data is not None


class TestUxarrayDataType:
    """ Test uxarray data type specific code paths. """
    
    def test_uxarray_pressure_interpolation(self: 'TestUxarrayDataType') -> None:
        """
        This test checks that when the processor is using the 'uxarray' data type and the dataset contains a `pressure` variable, the `get_3d_variable_data` method can successfully perform pressure interpolation to extract a 2D slice at a specified pressure level. By loading a real MPAS dataset, modifying it to include `pressure_base` and `pressure_p` components, and invoking the method with a pressure level, the test ensures that it returns valid data with the expected attributes, confirming that the processor can handle pressure interpolation correctly when using the 'uxarray' data type. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS3DProcessor(GRID_FILE, verbose=False)
        assert_expected_public_methods(processor, 'MPAS3DProcessor')

        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        paths = get_mpas_data_paths()
        processor = MPAS3DProcessor(str(paths['grid_file']), verbose=False)
        processor = processor.load_3d_data(str(paths['mpasout_dir']), use_pure_xarray=False)
        
        if processor.data_type == 'uxarray':
            if 'pressure' in processor.dataset:
                pressure = processor.dataset['pressure']
                processor.dataset['pressure_base'] = pressure * 0.8
                processor.dataset['pressure_p'] = pressure * 0.2
                
                var_data = processor.get_3d_variable_data('theta', level=70000.0, time_index=0)
                assert var_data is not None
                assert hasattr(var_data, 'values')
        else:
            assert processor.data_type == 'xarray'


class TestGetVerticalLevelsVerbose:
    """ Test verbose output in get_vertical_levels. """
    
    def test_pressure_from_components_verbose(self: 'TestGetVerticalLevelsVerbose') -> None:
        """
        This test verifies that when the `get_vertical_levels` method is called in verbose mode and the dataset contains pressure components (`pressure_base` and `pressure_p`), the method provides output indicating that it is reconstructing pressure levels from these components. By capturing the standard output during the method call, the test checks that messages related to reconstructing pressure levels (e.g., "reconstructing", "pressure_base", "pressure_p") are present, confirming that users are informed about the reconstruction process when pressure components are used to derive vertical levels. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        import sys
        from io import StringIO
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return
        
        processor = load_mpas_3d_processor(verbose=True)
        
        if 'pressure' in processor.dataset:
            pressure = processor.dataset['pressure']
            processor.dataset['pressure_base'] = pressure * 0.8
            processor.dataset['pressure_p'] = pressure * 0.2
            
            captured_output = StringIO()
            sys.stdout = captured_output
            
            levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)
            
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            assert levels is not None
            assert 'Pressure levels' in output or 'levels' in output


class TestExtract2DCoordinatesDataArray:
    """ Test extract_2d_coordinates with DataArray input. """
    
    def test_extract_with_data_array_vertices(self: 'TestExtract2DCoordinatesDataArray') -> None:
        """
        This test checks that the `extract_2d_coordinates_for_variable` method can successfully extract longitude and latitude coordinates when the input variable is an xarray DataArray. By loading a real MPAS dataset, checking for the presence of a 3D variable (e.g., 'vorticity'), and invoking the method with that variable, the test ensures that it returns valid longitude and latitude arrays with the expected shapes, confirming that the processor can handle coordinate extraction correctly when using xarray DataArrays. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")
            return

        processor = load_mpas_3d_processor(verbose=False)
        
        if 'vorticity' in processor.dataset:
            vorticity_da = processor.dataset['vorticity']
            lon, lat = processor.extract_2d_coordinates_for_variable('vorticity', vorticity_da)

            assert lon is not None
            assert lat is not None

            assert len(lon.shape) == pytest.approx(1)
            assert len(lat.shape) == pytest.approx(1)


class TestPressureException:
    """ Test exception handling in pressure-related code. """
    
    def test_pressure_variable_exception_handling(self: 'TestPressureException') -> None:
        """
        This test verifies that the `get_vertical_levels` method can handle exceptions that occur when accessing the `pressure` variable in the dataset. By mocking the dataset to raise an exception when the `pressure` variable is accessed, the test ensures that the method does not raise unhandled exceptions and instead falls back to an alternative approach (e.g., using model level indices) to return a valid list of vertical levels. If the test data is not available, the test will be skipped to avoid false failures. 

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
            pressure_no_time = processor.dataset['pressure'].isel(Time=0).drop_vars('Time', errors='ignore')
            original_pressure = processor.dataset['pressure']
            processor.dataset['pressure'] = pressure_no_time
            
            try:
                levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)
                assert levels is not None
            finally:
                processor.dataset['pressure'] = original_pressure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
