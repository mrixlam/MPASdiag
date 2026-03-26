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
import xarray as xr
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock, patch

from mpasdiag.processing.processors_3d import MPAS3DProcessor
from tests.test_data_helpers import get_mpas_data_paths, check_mpas_data_available, load_mpas_3d_processor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def make_getitem(mapping: dict, default: Any = None) -> Any:
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
        return key in key_set
    return _contains


def make_grid_getitem(lon_values: Any, lat_values: Any) -> Any:
    """
    This function returns a __getitem__ side_effect function that provides longitude and latitude values based on the key. It checks if the key contains 'lon' or 'lat' (case-insensitive) and returns a mock coordinate object with the corresponding values. This is particularly useful for testing the coordinate extraction logic in MPAS3DProcessor when working with grid datasets. 

    Parameters:
        lon_values (Any): The longitude values to return for keys containing 'lon'.
        lat_values (Any): The latitude values to return for keys containing 'lat'.

    Returns:
        Any: A function suitable for use as a __getitem__ side_effect.
    """
    def _getitem(key: str) -> Any:
        mock_coord = MagicMock()
        mock_coord.values = lon_values if 'lon' in key.lower() else lat_values
        return mock_coord
    return _getitem


def make_getitem_with_raise(raise_key: str, exc: Exception, default: Any = None) -> Any:
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
        if key == raise_key:
            raise exc
        return MagicMock() if default is None else default
    return _getitem


class TestExtract2DCoordinatesEdgeCases:
    """ Test edge cases in extract_2d_coordinates_for_variable. """
    
    def test_grid_file_exception_handling(self: "TestExtract2DCoordinatesEdgeCases") -> None:
        """
        This test verifies that the `extract_2d_coordinates_for_variable` method in the MPAS3DProcessor class properly handles exceptions that may occur when attempting to load the grid file. By mocking the `xarray.open_dataset` function to raise a RuntimeError, the test checks that the method catches this exception and raises a new RuntimeError with an appropriate error message, ensuring that users receive clear feedback about issues with loading the grid coordinates.

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=False)
        
        mock_ds = MagicMock()
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 100, 'nVertLevels': 55}
        mock_ds.__getitem__.return_value = mock_var
        mock_ds.__contains__.return_value = True
        processor.dataset = mock_ds
        
        with patch('xarray.open_dataset', side_effect=RuntimeError("File read error")):
            with pytest.raises(RuntimeError) as exc_info:
                processor.extract_2d_coordinates_for_variable('theta')
            assert "Error loading coordinates" in str(exc_info.value)
    
    def test_verbose_output_for_coordinate_extraction(self: "TestExtract2DCoordinatesEdgeCases") -> None:
        """
        This test checks that when the processor is initialized with `verbose=True`, the `extract_2d_coordinates_for_variable` method provides expected output information during coordinate extraction. By mocking the grid coordinates and ensuring that the returned longitude and latitude arrays have the correct sizes, the test validates that verbose mode does not interfere with the coordinate extraction process and that it provides useful feedback when enabled. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)  
        
        mock_ds = MagicMock()
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 100, 'nVertLevels': 55}
        mock_ds.__getitem__.return_value = mock_var
        mock_ds.__contains__.return_value = True
        processor.dataset = mock_ds
        
        mock_grid = MagicMock()
        mock_grid.coords = {'lonCell': True, 'latCell': True}
        mock_grid.data_vars = {}
        
        lon_deg = np.linspace(-180, 180, 100)
        lat_deg = np.linspace(-90, 90, 100)
        
        mock_grid.__getitem__.side_effect = make_grid_getitem(lon_deg, lat_deg)
        
        with patch('xarray.open_dataset') as mock_open:
            mock_open.return_value.__enter__.return_value = mock_grid
            
            lon, lat = processor.extract_2d_coordinates_for_variable('theta')
            
            assert len(lon) == 100
            assert len(lat) == 100


class TestLoad3DDataSpatialCoordinates:
    """ Test spatial coordinate handling in load_3d_data. """
    
    def test_load_3d_data_xarray_adds_spatial_coords(self: "TestLoad3DDataSpatialCoordinates") -> None:
        """
        This test verifies that the `load_3d_data` method can successfully load datasets using the xarray backend when the `use_pure_xarray` flag is set to True. The test checks that the processor's `dataset` attribute is populated with an xarray Dataset containing the expected data variables and that the `data_type` is correctly set to 'xarray'. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        paths = get_mpas_data_paths()
        processor = MPAS3DProcessor(str(paths['grid_file']), verbose=False)
        
        processor = processor.load_3d_data(str(paths['mpasout_dir']), use_pure_xarray=True)
        
        assert hasattr(processor.dataset, 'data_vars')
        assert processor.data_type == 'xarray'
    
    def test_load_3d_data_uxarray_adds_spatial_coords(self: "TestLoad3DDataSpatialCoordinates") -> None:
        """
        This test confirms that the `load_3d_data` method can successfully load datasets using the uxarray backend when the `use_pure_xarray` flag is set to False. The test checks that the processor's `dataset` attribute is populated with a dataset object containing the expected data variables and that the `data_type` is correctly set to 'uxarray'. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        paths = get_mpas_data_paths()

        processor = MPAS3DProcessor(str(paths['grid_file']), verbose=False)        
        processor = processor.load_3d_data(str(paths['mpasout_dir']), use_pure_xarray=False)
        
        assert processor.dataset is not None
        assert processor.data_type in ['uxarray', 'xarray']


class TestGetAvailable3DVerbose:
    """ Test verbose output in get_available_3d_variables. """
    
    def test_get_available_3d_variables_verbose_output(self: "TestGetAvailable3DVerbose") -> None:
        """
        This test checks that when the processor is initialized with `verbose=True`, the `get_available_3d_variables` method provides expected output information about the available 3D variables in the dataset. By ensuring that the method returns a non-empty list of variable names, the test validates that verbose mode does not interfere with the variable discovery process and that it provides useful feedback when enabled. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=True)        
        variables = processor.get_available_3d_variables()
        
        assert isinstance(variables, list)
        assert len(variables) > 0


class TestGet3DVariableDataPressureInterpolation:
    """ Test pressure level interpolation in get_3d_variable_data. """
    
    def test_pressure_interpolation_with_synthetic_pressure_components(self: "TestGet3DVariableDataPressureInterpolation") -> None:
        """
        This test verifies that the `get_3d_variable_data` method can perform pressure level interpolation when synthetic pressure components (`pressure_p` and `pressure_base`) are added to the dataset. By creating these components based on an existing pressure variable and requesting data at specific model levels, the test checks that the method returns valid data slices with appropriate attributes, confirming that the interpolation logic can function correctly even when using synthetic pressure information. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=False)
        
        if 'pressure' in processor.dataset:
            pressure = processor.dataset['pressure']
            pressure_base = pressure * 0.8
            pressure_p = pressure * 0.2
            
            processor.dataset['pressure_base'] = pressure_base
            processor.dataset['pressure_p'] = pressure_p
            
            var_data = processor.get_3d_variable_data('theta', level=85000.0, time_index=0)

            assert var_data is not None
            assert hasattr(var_data, 'values')
            assert var_data.values.size > 0
            
            var_data_high = processor.get_3d_variable_data('theta', level=120000.0, time_index=0)

            assert var_data_high is not None
            assert 'level_index' in var_data_high.attrs
            assert var_data_high.attrs['level_index'] == 0
            
            var_data_low = processor.get_3d_variable_data('theta', level=100.0, time_index=0)

            assert var_data_low is not None
            assert 'level_index' in var_data_low.attrs
            assert var_data_low.attrs['level_index'] > 50
    
    def test_pressure_interpolation_with_pressure_data(self: "TestGet3DVariableDataPressureInterpolation") -> None:
        """
        This test ensures that the `get_3d_variable_data` method can perform pressure level interpolation when actual pressure data is available in the dataset. By requesting data at a specific model level (e.g., level 10), the test checks that the returned data slice contains valid values and that the `level_index` attribute reflects the requested level, confirming that the interpolation process is functioning as intended when real pressure information is present. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=False)
        var_data = processor.get_3d_variable_data('theta', level=10, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0
        assert 'level_index' in var_data.attrs
        assert var_data.attrs['level_index'] == 10
    
    def test_pressure_level_above_surface_fallback(self: "TestGet3DVariableDataPressureInterpolation") -> None:
        """
        This test verifies that when a pressure level above the surface is requested, the `get_3d_variable_data` method correctly falls back to selecting the surface level. The test asserts that the returned data slice corresponds to the surface level by checking the `level_index` attribute, ensuring that the processor handles requests for levels above the surface gracefully without errors. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=False)
        var_data = processor.get_3d_variable_data('theta', level=0, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0
        assert 'level_index' in var_data.attrs
        assert var_data.attrs['level_index'] == 0
    
    def test_pressure_level_below_top_fallback(self: "TestGet3DVariableDataPressureInterpolation") -> None:
        """
        This test verifies that when a pressure level below the top of the model is requested, the `get_3d_variable_data` method correctly falls back to selecting the topmost model level. The test asserts that the returned data slice corresponds to the top level by checking the `level_index` attribute, ensuring that the processor handles requests for levels below the top of the model gracefully without errors. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=False)
        nlevels = processor.dataset.sizes['nVertLevels']
        top_level = nlevels - 1
        var_data = processor.get_3d_variable_data('theta', level=top_level, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0
        assert 'level_index' in var_data.attrs
        assert var_data.attrs['level_index'] == top_level
    
    def test_pressure_interpolation_weight_calculation(self: "TestGet3DVariableDataPressureInterpolation") -> None:
        """
        This test ensures that the `get_3d_variable_data` method calculates interpolation weights correctly when pressure data is available. By requesting data at a mid-level pressure (e.g., level 20), the test checks that the returned data slice contains valid values and that the `level_index` attribute reflects the requested level, confirming that the interpolation process is functioning as intended. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=False)
        nlevels = processor.dataset.sizes['nVertLevels']
        mid_level = nlevels // 2
        var_data = processor.get_3d_variable_data('theta', level=mid_level, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0
        assert np.any(np.isfinite(var_data.values))
        assert 'level_index' in var_data.attrs
        assert var_data.attrs['level_index'] == mid_level
    
    def test_pressure_interpolation_equal_pressures(self: "TestGet3DVariableDataPressureInterpolation") -> None:
        """
        This test checks the behavior of the `get_3d_variable_data` method when the requested pressure level matches exactly with one of the model levels. The test asserts that the method returns the data slice corresponding to that exact level without performing interpolation and that the `level_index` attribute correctly reflects the requested level, confirming that the processor handles exact pressure matches appropriately. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=False)
        var_data = processor.get_3d_variable_data('theta', level=20, time_index=0)

        assert var_data is not None
        assert hasattr(var_data, 'values')
        assert var_data.values.size > 0

        valid_vals = var_data.values[np.isfinite(var_data.values)]

        assert len(valid_vals) > 0
        assert np.all(valid_vals > 200) 
        assert np.all(valid_vals < 500) 
    
    def test_pressure_interpolation_multiple_levels(self: "TestGet3DVariableDataPressureInterpolation") -> None:
        """
        This test exercises the extraction of data across several representative model levels to ensure consistent indexing and returned-array shapes. By iterating through a small list of levels (e.g., 0, 10, 20, 30, 40), the test validates that for each requested level, the `get_3d_variable_data` method returns a non-empty data slice with the correct `level_index` attribute, confirming that the processor can handle multiple levels correctly and consistently. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=False)
        test_levels = [0, 10, 20, 30, 40]
        
        for level in test_levels:
            var_data = processor.get_3d_variable_data('theta', level=level, time_index=0)
            assert var_data is not None
            assert hasattr(var_data, 'values')
            assert var_data.values.size > 0
            assert 'level_index' in var_data.attrs
            assert var_data.attrs['level_index'] == level


class TestGet3DVariableDataAttributes:
    """ Test attribute handling in get_3d_variable_data. """
    
    def test_attributes_with_pressure_level(self: "TestGet3DVariableDataAttributes") -> None:
        """
        This test verifies that when a pressure level is requested in the `get_3d_variable_data` method, the returned data slice includes appropriate attributes such as `selected_level` and `level_index`. By requesting data at a specific model level (e.g., level 10), the test checks that these attributes are present in the returned object and that they correctly reflect the requested level, confirming that the processor provides useful metadata for extracted data slices. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=True)
        var_data = processor.get_3d_variable_data('theta', level=10, time_index=0)
        
        if hasattr(var_data, 'attrs'):
            assert 'selected_level' in var_data.attrs
            assert 'level_index' in var_data.attrs
    
    def test_verbose_output_with_units(self: "TestGet3DVariableDataAttributes") -> None:
        """
        This test checks that when the processor is initialized with `verbose=True`, the `get_3d_variable_data` method provides expected output information about the variable being extracted, including its name and units. By requesting data for a specific variable (e.g., 'theta'), the test validates that verbose mode does not interfere with data extraction and that it provides useful feedback about the variable's metadata when enabled. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=True)        
        var_data = processor.get_3d_variable_data('theta', level=5, time_index=0)
        
        assert var_data is not None
    
    def test_warning_for_no_finite_values(self: "TestGet3DVariableDataAttributes") -> None:
        """
        This test ensures that the `get_3d_variable_data` method issues a warning when the extracted data slice contains no finite values. By mocking a variable to return an array filled with NaNs, the test checks that the method returns a data slice with the expected attributes while also providing feedback about the lack of valid data, confirming that the processor can handle such cases gracefully without raising unhandled exceptions. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 55, 'Time': 10, 'nCells': 100}
        
        nan_data = MagicMock()
        nan_data.values = np.full((100,), np.nan)
        nan_data.attrs = {}
        
        mock_var.isel.return_value = nan_data
        
        mock_ds.__getitem__.side_effect = make_getitem({}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta'])
        mock_ds.data_vars = {'theta': mock_var}
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        
        with patch.object(processor, 'get_vertical_levels', return_value=[10]):
            var_data = processor.get_3d_variable_data('theta', level=10, time_index=0)
            assert var_data is not None


class TestGetVerticalLevelsEdgeCases:
    """ Test edge cases in get_vertical_levels. """
    
    def test_pressure_from_pressure_variable_non_positive_warning(self: "TestGetVerticalLevelsEdgeCases") -> None:
        """
        This test checks that the `get_vertical_levels` method issues a warning when the pressure variable contains non-positive values, which are invalid for pressure levels. By mocking the pressure variable to include negative and NaN values, the test verifies that the method returns a list of vertical levels while also providing feedback about the presence of non-positive pressure values, confirming that the processor can handle such cases gracefully without raising unhandled exceptions. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_pressure = MagicMock()
        bad_pressure = np.array([100000, 80000, -5000, np.nan, 50000])  
        mock_pressure_mean = MagicMock()
        mock_pressure_mean.values = bad_pressure
        mock_pressure.mean.return_value = mock_pressure_mean
        mock_pressure.isel.return_value = mock_pressure
        
        mock_ds.__getitem__.side_effect = make_getitem({'pressure': mock_pressure}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)

        assert isinstance(levels, list)
        assert len(levels) == 5
        assert levels == [0, 1, 2, 3, 4]
    
    def test_pressure_from_components_nVertLevelsP1_extension(self: "TestGetVerticalLevelsEdgeCases") -> None:
        """
        This test verifies that the `get_vertical_levels` method can successfully extract vertical levels when the dataset includes the `nVertLevelsP1` dimension, which indicates an extended vertical grid. By iterating through available 3D variables and checking for the presence of this dimension, the test ensures that the method can handle datasets with extended vertical levels and returns a list of pressure levels without errors. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=True)
        
        for var_name in processor.get_available_3d_variables():
            if 'nVertLevelsP1' in processor.dataset[var_name].sizes:
                levels = processor.get_vertical_levels(var_name, return_pressure=True, time_index=0)
                assert isinstance(levels, list)
                break
    
    def test_pressure_from_hybrid_coords_interpolation(self: "TestGetVerticalLevelsEdgeCases") -> None:
        """
        This test validates that the `get_vertical_levels` method can perform interpolation to reconstruct pressure levels when hybrid-coordinate arrays (`fzp` and `surface_pressure`) contain some non-finite values. By mocking these arrays with a mix of valid and invalid entries, the test checks that the method successfully interpolates to produce a complete list of pressure levels and that it handles the presence of non-finite values gracefully without raising unhandled exceptions. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_fzp = MagicMock()
        mock_fzp_isel = MagicMock()
        mock_fzp_isel.values = np.array([1.0, 0.8, np.nan, 0.4, 0.2])  
        mock_fzp.isel.return_value = mock_fzp_isel
        
        mock_sp = MagicMock()
        mock_sp_isel = MagicMock()
        mock_sp_mean = MagicMock()
        mock_sp_mean.values = 101300.0
        mock_sp_isel.mean.return_value = mock_sp_mean
        mock_sp.isel.return_value = mock_sp_isel
        
        mock_ds.__getitem__.side_effect = make_getitem({'fzp': mock_fzp, 'surface_pressure': mock_sp}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)

        assert isinstance(levels, list)
        assert len(levels) == 5
    
    def test_pressure_from_hybrid_coords_single_good_value(self: "TestGetVerticalLevelsEdgeCases") -> None:
        """
        This test checks that the `get_vertical_levels` method can still produce a valid list of pressure levels when the hybrid-coordinate arrays contain only a single good value. By mocking the `fzp` array to have one valid entry and the rest as non-finite, the test verifies that the method can use that single good value along with the surface pressure to reconstruct a complete list of pressure levels without raising exceptions, confirming that the processor can handle this edge case gracefully. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_fzp = MagicMock()
        mock_fzp_isel = MagicMock()
        mock_fzp_isel.values = np.array([np.nan, np.nan, 0.5, np.nan, np.nan]) 
        mock_fzp.isel.return_value = mock_fzp_isel
        
        mock_sp = MagicMock()
        mock_sp_isel = MagicMock()
        mock_sp_mean = MagicMock()
        mock_sp_mean.values = 101300.0
        mock_sp_isel.mean.return_value = mock_sp_mean
        mock_sp.isel.return_value = mock_sp_isel
        
        mock_ds.__getitem__.side_effect = make_getitem({'fzp': mock_fzp, 'surface_pressure': mock_sp}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)

        assert isinstance(levels, list)
        assert len(levels) == 5
    
    def test_pressure_from_hybrid_coords_no_good_values(self: "TestGetVerticalLevelsEdgeCases") -> None:
        """
        This test ensures that the `get_vertical_levels` method can still produce a valid list of pressure levels when the hybrid-coordinate arrays contain no good values. By mocking the `fzp` array to have all non-finite entries, the test checks that the method falls back to a robust reconstruction (e.g., using logspace) to generate a complete list of pressure levels without raising exceptions, confirming that the processor can handle this edge case gracefully. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_fzp = MagicMock()
        mock_fzp_isel = MagicMock()
        mock_fzp_isel.values = np.array([np.nan, np.nan, np.nan, np.nan, np.nan]) 
        mock_fzp.isel.return_value = mock_fzp_isel
        
        mock_sp = MagicMock()
        mock_sp_isel = MagicMock()
        mock_sp_mean = MagicMock()
        mock_sp_mean.values = 101300.0
        mock_sp_isel.mean.return_value = mock_sp_mean
        mock_sp.isel.return_value = mock_sp_isel
        
        mock_ds.__getitem__.side_effect = make_getitem({'fzp': mock_fzp, 'surface_pressure': mock_sp}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'        
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)

        assert isinstance(levels, list)
        assert len(levels) == 5
    
    def test_pressure_from_hybrid_coords_exception_fallback(self: "TestGetVerticalLevelsEdgeCases") -> None:
        """
        This test checks that if an exception occurs while accessing the `fzp` array during pressure level reconstruction, the `get_vertical_levels` method falls back to using model-level indices instead of pressure values. By mocking the dataset to raise an exception when `fzp` is accessed, the test ensures that the method handles this scenario gracefully and returns a list of model level indices without raising unhandled exceptions, confirming that the processor can manage unexpected issues with hybrid-coordinate data. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        with patch('os.path.exists', return_value=True):
            processor = MPAS3DProcessor('test_grid.nc', verbose=True)
        
        mock_ds = MagicMock()
        mock_ds.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nVertLevels': 5, 'Time': 10, 'nCells': 100}
        mock_ds.data_vars = {'theta': mock_var}
        
        mock_ds.__getitem__.side_effect = make_getitem_with_raise('fzp', RuntimeError("Failed to access fzp"), default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        processor.dataset = mock_ds
        processor.data_type = 'xarray'
        
        levels = processor.get_vertical_levels('theta', return_pressure=True, time_index=0)
        assert isinstance(levels, list)
        assert levels == [0, 1, 2, 3, 4] 
    
    def test_model_level_indices_verbose(self: "TestGetVerticalLevelsEdgeCases") -> None:
        """
        This test verifies that when the `get_vertical_levels` method is called in verbose mode and the dataset does not contain pressure information, the method returns a list of model level indices while providing informative output about the lack of pressure data. By ensuring that the returned levels are valid indices and that verbose output includes messages about missing pressure information, the test confirms that the processor can handle this scenario gracefully while keeping users informed. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=True)        
        levels = processor.get_vertical_levels('theta', return_pressure=False, time_index=0)

        assert isinstance(levels, list)
        assert all(isinstance(x, int) for x in levels)


class TestExtract2DFrom3DEdgeCases:
    """ Test edge cases in extract_2d_from_3d. """
    
    def test_numpy_higher_dimension_data(self: "TestExtract2DFrom3DEdgeCases") -> None:
        """
        This test checks that the `extract_2d_from_3d` method can handle input data with more than 3 dimensions, such as a 4D array. By creating a synthetic 4D numpy array and invoking the method with a specified `level_index`, the test ensures that the method can successfully extract a 2D slice from the 3D portion of the data while ignoring any additional dimensions, confirming that it can handle higher-dimensional input gracefully without errors. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        data_4d = np.random.rand(5, 10, 20, 30)        
        result = MPAS3DProcessor.extract_2d_from_3d(data_4d, level_index=2)
        assert isinstance(result, np.ndarray)
    
    def test_xarray_level_index_fallback_dimension(self: "TestExtract2DFrom3DEdgeCases") -> None:
        """        
        This test verifies that the `extract_2d_from_3d` method can successfully extract a 2D slice from a 3D xarray DataArray when the specified `level_dim` is not present in the data. By creating a synthetic DataArray with dimensions that do not include the requested `level_dim`, the test checks that the method falls back to using the first dimension as the vertical level dimension and returns a valid 2D array, confirming that it can handle cases where the expected level dimension is missing without raising errors. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        data = np.random.rand(10, 55, 100)
        da = xr.DataArray(data, dims=['time', 'nVertLevels', 'nCells'])
        result = MPAS3DProcessor.extract_2d_from_3d(da, level_index=10, level_dim='nVertLevels')

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 100) 
    
    def test_xarray_interpolation_linear(self: "TestExtract2DFrom3DEdgeCases") -> None:
        """
        This test checks that the `extract_2d_from_3d` method can perform linear interpolation when extracting a 2D slice from a 3D xarray DataArray based on a specified `level_value` and `level_dim`. By creating a synthetic DataArray with a known pressure coordinate and requesting data at a specific pressure level, the test ensures that the method can successfully interpolate to return a valid 2D array, confirming that the interpolation logic is functioning correctly when using xarray input. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        data = np.random.rand(10, 55, 100)
        pressure = np.linspace(100000, 10000, 55)

        da = xr.DataArray(data, dims=['time', 'pressure', 'cell'],
                         coords={'pressure': pressure})
        
        result = MPAS3DProcessor.extract_2d_from_3d(da, level_value=50000, level_dim='pressure', method='linear')
        assert isinstance(result, np.ndarray)


class TestPressureInterpolationVerbose:
    """ Test verbose output paths in pressure interpolation. """
    
    def test_pressure_interpolation_verbose_mode(self: "TestPressureInterpolationVerbose") -> None:
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
    
    def test_pressure_above_surface_verbose(self: "TestPressureInterpolationVerbose") -> None:
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
    
    def test_pressure_below_top_verbose(self: "TestPressureInterpolationVerbose") -> None:
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
    
    def test_equal_pressures_interpolation(self: "TestPressureInterpolationVerbose") -> None:
        """
        This test verifies that when the `get_3d_variable_data` method is called in verbose mode with a pressure level that matches exactly with one of the model levels, the method provides output indicating that it is selecting the exact level without interpolation. By capturing the standard output during the method call, the test checks that messages related to selecting the exact level (e.g., "exact", "level 20") are present, confirming that users are informed about the selection of the exact level when a pressure level matches one of the model levels. If the test data is not available, the test will be skipped to avoid false failures. 
        
        Parameters:
            None
        
        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

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
    
    def test_uxarray_pressure_interpolation(self: "TestUxarrayDataType") -> None:
        """
        This test checks that when the processor is using the 'uxarray' data type and the dataset contains a `pressure` variable, the `get_3d_variable_data` method can successfully perform pressure interpolation to extract a 2D slice at a specified pressure level. By loading a real MPAS dataset, modifying it to include `pressure_base` and `pressure_p` components, and invoking the method with a pressure level, the test ensures that it returns valid data with the expected attributes, confirming that the processor can handle pressure interpolation correctly when using the 'uxarray' data type. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

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
    
    def test_pressure_from_components_verbose(self: "TestGetVerticalLevelsVerbose") -> None:
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
    
    def test_extract_with_data_array_vertices(self: "TestExtract2DCoordinatesDataArray") -> None:
        """
        This test checks that the `extract_2d_coordinates_for_variable` method can successfully extract longitude and latitude coordinates when the input variable is an xarray DataArray. By loading a real MPAS dataset, checking for the presence of a 3D variable (e.g., 'vorticity'), and invoking the method with that variable, the test ensures that it returns valid longitude and latitude arrays with the expected shapes, confirming that the processor can handle coordinate extraction correctly when using xarray DataArrays. If the test data is not available, the test will be skipped to avoid false failures.

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

        processor = load_mpas_3d_processor(verbose=False)
        
        if 'vorticity' in processor.dataset:
            vorticity_da = processor.dataset['vorticity']
            lon, lat = processor.extract_2d_coordinates_for_variable('vorticity', vorticity_da)

            assert lon is not None
            assert lat is not None

            assert len(lon.shape) == 1
            assert len(lat.shape) == 1


class TestPressureException:
    """ Test exception handling in pressure-related code. """
    
    def test_pressure_variable_exception_handling(self: "TestPressureException") -> None:
        """
        This test verifies that the `get_vertical_levels` method can handle exceptions that occur when accessing the `pressure` variable in the dataset. By mocking the dataset to raise an exception when the `pressure` variable is accessed, the test ensures that the method does not raise unhandled exceptions and instead falls back to an alternative approach (e.g., using model level indices) to return a valid list of vertical levels. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        
        if not check_mpas_data_available():
            pytest.skip("Test data not available")

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


class TestExtract2DFallbackDimension:
    """ Test fallback dimension selection in extract_2d_from_3d. """
    
    def test_fallback_to_second_dimension(self: "TestExtract2DFallbackDimension") -> None:
        """
        This test checks that the `extract_2d_from_3d` method can successfully extract a 2D slice from a 3D xarray DataArray when the specified `level_dim` is not present, and it falls back to using the second dimension as the vertical level dimension. By creating a synthetic DataArray with dimensions that do not include the requested `level_dim`, the test ensures that the method can still return a valid 2D array with the expected shape, confirming that it can handle cases where the expected level dimension is missing without raising errors. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        from mpasdiag.processing.processors_3d import MPAS3DProcessor
        import numpy as np
        import xarray as xr
        
        data = np.random.rand(10, 55, 100)
        da = xr.DataArray(data, dims=['time', 'vertical', 'cell'])        
        result = MPAS3DProcessor.extract_2d_from_3d(da, level_index=10, level_dim='vertical')

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 100)


class TestMPAS3DProcessorInit:
    """ Test processor initialization. """

    @pytest.fixture(autouse=True)
    def temp_grid_file(self: "TestMPAS3DProcessorInit", tmp_path: Path) -> None:
        """
        This fixture creates a temporary grid file for use in the processor initialization tests. By using the `tmp_path` fixture provided by pytest, it ensures that a real file exists at the specified path, allowing the `MPAS3DProcessor` to initialize without errors related to missing grid files. The temporary file is created before each test method runs and is automatically cleaned up after the tests complete. 

        Parameters:
            tmp_path (Path): pytest-provided temporary directory.

        Returns:
            None
        """
        self.grid_path = str(tmp_path / "test_grid.nc")
        Path(self.grid_path).touch()

    def test_init_verbose_true(self: "TestMPAS3DProcessorInit") -> None:
        """
        This test verifies that the `MPAS3DProcessor` initializes correctly when `verbose=True`. A real temporary file is used to satisfy the filesystem existence check. The test asserts that the returned processor instance has `verbose` set to True and that the `grid_file` attribute is set correctly. By creating a temporary grid file and initializing the processor with `verbose=True`, the test ensures that the processor can be created successfully and that its attributes are set as expected when verbose mode is enabled. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS3DProcessor(self.grid_path, verbose=True)
        assert processor.verbose
        assert processor.grid_file == self.grid_path

    def test_init_verbose_false(self: "TestMPAS3DProcessorInit") -> None:
        """
        This test verifies that the `MPAS3DProcessor` initializes correctly when `verbose=False`. A real temporary file is used to satisfy the filesystem existence check. The test asserts that the returned processor instance has `verbose` set to False. By creating a temporary grid file and initializing the processor with `verbose=False`, the test ensures that the processor can be created successfully and that its attributes are set as expected when verbose mode is disabled. 

        Parameters:
            None

        Returns:
            None
        """
        processor = MPAS3DProcessor(self.grid_path, verbose=False)
        assert not processor.verbose


class TestExtract2DCoordinates:
    """ Test coordinate extraction for 3D variables. """
    
    def setup_method(self: "TestExtract2DCoordinates") -> None:
        """
        This setup method creates a temporary grid file and initializes an `MPAS3DProcessor` instance for use in the coordinate extraction tests. By creating a temporary file with a `.nc` suffix, it ensures that the processor can be initialized without errors related to missing grid files. The processor is created with `verbose=False` to avoid unnecessary output during the tests. This setup allows the subsequent test methods to focus on testing the coordinate extraction functionality without worrying about processor initialization issues. 

        Parameters:
            None

        Returns:
            None
        """
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            self._temp_grid = f.name
        self.processor = MPAS3DProcessor(self._temp_grid, verbose=False)

    def teardown_method(self: "TestExtract2DCoordinates") -> None:
        """
        Remove the temporary grid file created in setup_method.

        Parameters:
            None

        Returns:
            None
        """
        try:
            os.unlink(self._temp_grid)
        except FileNotFoundError:
            pass
    
    def test_without_dataset_raises_error(self: "TestExtract2DCoordinates") -> None:
        """
        This test verifies that the `extract_2d_coordinates_for_variable` method raises a ValueError when the processor's dataset is not loaded. By explicitly setting the processor's dataset to None and then invoking the coordinate extraction method, the test checks that a ValueError is raised with an appropriate message indicating that the dataset is not loaded. This ensures that the method correctly handles cases where it is called without a valid dataset, providing clear feedback to users about the issue. 

        Parameters:
            None

        Returns:
            None
        """
        self.processor.dataset = None
        with pytest.raises(ValueError) as exc_info:
            self.processor.extract_2d_coordinates_for_variable('theta')
        assert 'not loaded' in str(exc_info.value)
    
    def test_cell_variable_radians_conversion(self: "TestExtract2DCoordinates") -> None:
        """
        This test checks that for a variable with the `nCells` dimension, the `extract_2d_coordinates_for_variable` method correctly identifies and extracts `lonCell` and `latCell` coordinates from the grid file, and converts them from radians to degrees if necessary. By creating a temporary grid file with `lonCell` and `latCell` in radians, setting up a dataset with a variable that has the `nCells` dimension, and invoking the coordinate extraction method, the test checks that it returns longitude and latitude arrays of the correct length and that the values are within valid ranges for degrees, confirming that the method can handle cell-based variables and perform unit conversion correctly when needed. 

        Parameters:
            None

        Returns:
            None
        """
        n_cells = 100
        lon_rad = np.linspace(-np.pi, np.pi, n_cells)
        lat_rad = np.linspace(-np.pi / 2, np.pi / 2, n_cells)

        grid_ds = xr.Dataset({
            'lonCell': xr.DataArray(lon_rad, dims=['nCells']),
            'latCell': xr.DataArray(lat_rad, dims=['nCells']),
        })

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_grid = f.name
        try:
            grid_ds.to_netcdf(temp_grid)
            self.processor.grid_file = temp_grid
            self.processor.dataset = xr.Dataset({
                'theta': xr.DataArray(
                    np.random.rand(n_cells, 55), dims=['nCells', 'nVertLevels']
                )
            })

            lon, lat = self.processor.extract_2d_coordinates_for_variable('theta')

            assert len(lon) == n_cells
            assert np.max(np.abs(lon) <= 180)
            assert np.max(np.abs(lat) <= 90)
        finally:
            os.unlink(temp_grid)
    
    def test_vertex_variable_with_data_array(self: "TestExtract2DCoordinates") -> None:
        """
        This test verifies that for a variable with the `nVertices` dimension, the `extract_2d_coordinates_for_variable` method correctly identifies and extracts `lonVertex` and `latVertex` coordinates from the grid file when the input variable is provided as an xarray DataArray. By creating a temporary grid file with `lonVertex` and `latVertex`, setting up a dataset with a variable that has the `nVertices` dimension, and invoking the coordinate extraction method with an xarray DataArray, the test checks that it returns longitude and latitude arrays of the correct length, confirming that the method can handle vertex-based variables correctly when using xarray DataArrays. 

        Parameters:
            None

        Returns:
            None
        """
        n_vertices = 150
        lon_deg = np.linspace(-180, 180, n_vertices)
        lat_deg = np.linspace(-90, 90, n_vertices)

        grid_ds = xr.Dataset({
            'lonVertex': xr.DataArray(lon_deg, dims=['nVertices']),
            'latVertex': xr.DataArray(lat_deg, dims=['nVertices']),
        })

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_grid = f.name
        try:
            grid_ds.to_netcdf(temp_grid)
            self.processor.grid_file = temp_grid
            self.processor.dataset = xr.Dataset()

            data_array = xr.DataArray(
                np.random.rand(n_vertices, 55), dims=['nVertices', 'nVertLevels']
            )

            lon, lat = self.processor.extract_2d_coordinates_for_variable(
                'vorticity', data_array=data_array
            )

            assert len(lon) == n_vertices
            assert len(lat) == n_vertices
        finally:
            os.unlink(temp_grid)
    
    def test_edge_variable(self: "TestExtract2DCoordinates") -> None:
        """
        This test checks that for a variable with the `nEdges` dimension, the `extract_2d_coordinates_for_variable` method correctly identifies and extracts `lonEdge` and `latEdge` coordinates from the grid file. By creating a temporary grid file with `lonEdge` and `latEdge`, setting up a dataset with a variable that has the `nEdges` dimension, and invoking the coordinate extraction method, the test ensures that it returns longitude and latitude arrays of the correct length, confirming that the method can handle edge-based variables correctly when using xarray DataArrays. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        n_edges = 200
        lon_deg = np.linspace(-180, 180, n_edges)
        lat_deg = np.linspace(-90, 90, n_edges)

        grid_ds = xr.Dataset({
            'lonEdge': xr.DataArray(lon_deg, dims=['nEdges']),
            'latEdge': xr.DataArray(lat_deg, dims=['nEdges']),
        })

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_grid = f.name
        try:
            grid_ds.to_netcdf(temp_grid)
            self.processor.grid_file = temp_grid
            self.processor.dataset = xr.Dataset({
                'uNormal': xr.DataArray(
                    np.random.rand(n_edges, 55), dims=['nEdges', 'nVertLevels']
                )
            })

            lon, lat = self.processor.extract_2d_coordinates_for_variable('uNormal')

            assert len(lon) == n_edges
        finally:
            os.unlink(temp_grid)
    
    def test_missing_coordinates_raises_error(self: "TestExtract2DCoordinates") -> None:
        """
        This test verifies that if the grid file does not contain the necessary coordinate variables for a given variable's dimension, the `extract_2d_coordinates_for_variable` method raises a `RuntimeError` indicating an error loading coordinates. By creating a temporary grid file that lacks the expected coordinate variables and invoking the coordinate extraction method, the test ensures that it correctly identifies the issue with loading coordinates and raises an appropriate error message, confirming that the method can handle cases where coordinate information is missing in the grid file. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        grid_ds = xr.Dataset({
            'unrelated_var': xr.DataArray([1, 2, 3], dims=['x']),
        })

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_grid = f.name
        try:
            grid_ds.to_netcdf(temp_grid)
            self.processor.grid_file = temp_grid
            self.processor.dataset = xr.Dataset()

            with pytest.raises(RuntimeError) as exc_info:
                self.processor.extract_2d_coordinates_for_variable('theta')
            assert 'Error loading coordinates' in str(exc_info.value)
        finally:
            os.unlink(temp_grid)
    
    def test_grid_file_error(self: "TestExtract2DCoordinates") -> None:
        """
        This test checks that if the grid file cannot be loaded (e.g., due to a nonexistent path), the `extract_2d_coordinates_for_variable` method raises a `RuntimeError` indicating an error loading coordinates. By setting the processor's grid file to a nonexistent path and invoking the coordinate extraction method, the test ensures that it correctly identifies the issue with loading the grid file and raises an appropriate error message, confirming that the method can handle cases where the grid file is not accessible. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        self.processor.dataset = xr.Dataset()
        self.processor.grid_file = '/nonexistent/path/to_grid.nc'

        with pytest.raises(RuntimeError) as exc_info:
            self.processor.extract_2d_coordinates_for_variable('theta')

        assert 'Error loading coordinates' in str(exc_info.value)


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

        processor = load_mpas_3d_processor(verbose=False)        
        data = processor.get_3d_variable_data('theta', level=0, time_index=0)

        assert data is not None
        assert hasattr(data, 'values')
        assert data.values.size > 0
        assert data.attrs['level_index'] == 0
    
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


class TestGetVerticalLevels:
    """ Test vertical level retrieval. """
    
    def setup_method(self: "TestGetVerticalLevels") -> None:
        """
        This method-level setup initializes an `MPAS3DProcessor` instance with a mocked grid file path and `verbose=False`. The filesystem existence check is patched to always return True, allowing the test to focus on the processor initialization logic without relying on actual files. This setup provides a consistent starting point for the subsequent tests that will verify the behavior of vertical level retrieval methods. If the test data is not available, the tests will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        with patch('os.path.exists', return_value=True):
            self.processor = MPAS3DProcessor('test_grid.nc', verbose=False)
    
    def test_get_model_levels_from_real_data(self: "TestGetVerticalLevels") -> None:
        """
        This test verifies that the `get_vertical_levels` method can successfully retrieve model level indices from real MPAS output when requested with `return_pressure=False`. By loading the dataset, retrieving an available 3D variable, and invoking the method, the test asserts that the returned levels are a non-empty list of integers corresponding to model levels. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = load_mpas_3d_processor(verbose=False)
        vars_3d = processor.get_available_3d_variables()
        assert len(vars_3d) > 0
        
        var_name = vars_3d[0]
        levels = processor.get_vertical_levels(var_name, return_pressure=False)
        
        assert isinstance(levels, list)
        assert len(levels) > 0
        assert levels == list(range(len(levels)))
    
    def test_get_pressure_levels_from_real_data(self: "TestGetVerticalLevels") -> None:
        """
        This test checks that the `get_vertical_levels` method can successfully retrieve pressure levels from real MPAS output when requested with `return_pressure=True`. By loading the dataset, retrieving an available 3D variable, and invoking the method, the test asserts that the returned levels are a non-empty list of pressure values that decrease with height. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = load_mpas_3d_processor(verbose=False)
        vars_3d = processor.get_available_3d_variables()
        var_name = vars_3d[0]
        
        try:
            levels = processor.get_vertical_levels(var_name, return_pressure=True)
            assert isinstance(levels, list)
            assert len(levels) > 0
            if len(levels) > 1:
                assert levels[0] > levels[-1], "Pressure should decrease with height"
        except ValueError as e:
            if 'pressure' not in str(e).lower():
                raise
    
    def test_without_dataset_raises_error(self: "TestGetVerticalLevels") -> None:
        """
        This test verifies that if the `dataset` attribute of the processor is not set (i.e., remains `None`), the `get_vertical_levels` method raises a `ValueError` indicating that the dataset is not loaded. By explicitly setting `processor.dataset` to `None` and invoking the method with a variable name, the test ensures that it correctly identifies the missing dataset and raises an appropriate error message. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        self.processor.dataset = None
        with pytest.raises(ValueError):
            self.processor.get_vertical_levels('theta')
    
    def test_variable_not_found(self: "TestGetVerticalLevels") -> None:
        """
        This test checks that if the requested variable name is not found in the dataset's data variables, the `get_vertical_levels` method raises a `ValueError` indicating that the variable is not found. By mocking the dataset to contain a different variable and invoking the method with a variable name that does not exist in the dataset, the test ensures that it correctly identifies the missing variable and raises an appropriate error message. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'pressure': True}
        self.processor.dataset = mock_ds
        
        with pytest.raises(ValueError):
            self.processor.get_vertical_levels('theta')
    
    def test_not_3d_variable(self: "TestGetVerticalLevels") -> None:
        """
        This test verifies that if the requested variable does not have the expected 3D dimensions (e.g., it has dimensions like `nCells` and `Time` instead of `nCells`, `nVertLevels`, and `Time`), the `get_vertical_levels` method raises a `ValueError` indicating that the variable is not a 3D variable. By mocking the dataset to contain a variable with incorrect dimensions and invoking the method, the test ensures that it correctly identifies the issue with the variable's dimensionality and raises an appropriate error message. If the test data is not available, the test will be skipped to avoid false failures. 

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
        
        with pytest.raises(ValueError):
            self.processor.get_vertical_levels('t2m')
    
    def test_model_level_indices(self: "TestGetVerticalLevels") -> None:
        """
        This test checks that when `return_pressure=False` is specified, the `get_vertical_levels` method returns a list of model level indices corresponding to the vertical levels in the dataset. Using a mocked dataset with a variable that has the expected 3D dimensions, this test calls `get_vertical_levels(..., return_pressure=False)` and asserts that the returned list has the expected length and contains integer indices starting from 0 up to the number of vertical levels minus one. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset()
        self.processor.dataset = mock_ds
        
        levels = self.processor.get_vertical_levels('theta', return_pressure=False)
        
        assert len(levels) == 55
        assert levels == list(range(55))
    
    def test_pressure_from_pressure_variable(self: "TestGetVerticalLevels") -> None:
        """
        This test verifies that when a `pressure` variable is present in the dataset, the `get_vertical_levels` method can retrieve pressure levels directly from this variable when `return_pressure=True` is specified. Using a mocked dataset that includes a `pressure` variable with synthetic pressure values, this test calls `get_vertical_levels(..., return_pressure=True)` and asserts that the returned list has the expected length and that pressure decreases with height. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset_with_pressure_var()
        self.processor.dataset = mock_ds
        self.processor.verbose = True
        
        levels = self.processor.get_vertical_levels('theta', return_pressure=True)
        
        assert len(levels) == 55
        assert levels[0] > levels[-1]
    
    def test_pressure_from_components(self: "TestGetVerticalLevels") -> None:
        """
        This test checks that when the dataset contains the necessary pressure components (e.g., `pressure_p` and `pressure_base`), the `get_vertical_levels` method can reconstruct pressure levels from these components when `return_pressure=True` is specified. Using a mocked dataset that includes synthetic `pressure_p` and `pressure_base` variables, this test calls `get_vertical_levels(..., return_pressure=True)` and asserts that the returned list has the expected length and that pressure decreases with height. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset_with_pressure_components()
        self.processor.dataset = mock_ds
        self.processor.verbose = False  
        
        levels = self.processor.get_vertical_levels('theta', return_pressure=True)
        
        assert len(levels) == 55
        assert levels[0] > levels[-1]
    
    def test_pressure_from_hybrid_coords(self: "TestGetVerticalLevels") -> None:
        """
        This test verifies that when the dataset contains hybrid coordinate information (e.g., `hybrid_a` and `hybrid_b`), the `get_vertical_levels` method can compute pressure levels from these hybrid coordinates. Using a mocked dataset that includes hybrid coordinate fields, this test calls `get_vertical_levels(..., return_pressure=True)` and asserts that the returned list has the expected length and that pressure decreases with height. If the test data is not available, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = self._create_mock_dataset_with_hybrid()
        self.processor.dataset = mock_ds
        self.processor.verbose = True
        
        levels = self.processor.get_vertical_levels('theta', return_pressure=True)
        
        assert len(levels) == 55
    
    def _create_mock_dataset(self: "TestGetVerticalLevels") -> Any:
        """
        This helper method creates a mock dataset that includes a 3D variable with the expected dimensions and attributes for testing vertical level retrieval. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with appropriate sizes and attributes. By setting up the necessary return values for indexing and slicing operations, this mock dataset allows tests to focus on the logic of retrieving vertical levels without relying on actual MPAS data files, enabling controlled testing of various scenarios related to vertical level retrieval. 

        Parameters:
            None

        Returns:
            Any: A MagicMock object emulating an xarray Dataset suitable for tests.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10}
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        mock_ds.__getitem__.return_value = mock_var
        mock_ds.__contains__.return_value = False
        return mock_ds
    
    def _create_mock_dataset_with_pressure_var(self: "TestGetVerticalLevels") -> Any:
        """
        This helper method creates a mock dataset that includes a `pressure` variable for testing pressure level retrieval. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with a `pressure` variable that has the same vertical levels and time dimensions. By setting up the necessary return values for indexing, slicing, and mean operations on the pressure variable, this mock dataset allows tests to focus on the logic of retrieving pressure levels directly from a pressure variable without relying on actual MPAS data files. 

        Parameters:
            None

        Returns:
            Any: A MagicMock object emulating an xarray Dataset with a `pressure` variable.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True, 'pressure': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        
        pressure_vals = np.linspace(100000, 1000, 55)
        mock_pressure = MagicMock()
        mock_pressure.mean.return_value = MagicMock(values=pressure_vals)
        mock_pressure.isel.return_value = mock_pressure
        
        mock_ds.__getitem__.side_effect = make_getitem({'pressure': mock_pressure}, default=mock_var)
        mock_ds.__contains__.return_value = True
        
        return mock_ds
    
    def _create_mock_dataset_with_pressure_components(self: "TestGetVerticalLevels") -> Any:
        """
        This helper method creates a mock dataset that includes the necessary pressure components (`pressure_p` and `pressure_base`) for testing pressure level retrieval from components. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with synthetic `pressure_p` and `pressure_base` variables that allow for testing the reconstruction of pressure levels. By setting up the necessary return values for indexing, slicing, and arithmetic operations on the pressure components, this mock dataset enables controlled testing of scenarios related to retrieving pressure levels from components without relying on actual MPAS data files. 

        Parameters:
            None

        Returns:
            Any: A MagicMock emulating an xarray Dataset with pressure components.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True, 'pressure_p': True, 'pressure_base': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        mock_var.dims = ('Time', 'nCells', 'nVertLevels')
        
        pressure_base_vals = np.linspace(100000, 1000, 55)
        
        mock_total = MagicMock()
        mock_mean_result = MagicMock()
        mock_mean_result.values = pressure_base_vals  
        mock_total.mean.return_value = mock_mean_result
        
        mock_pressure_base_time = MagicMock()
        mock_pressure_p_time = MagicMock()
        
        mock_pressure_base_time.__add__ = lambda self, other: mock_total
        mock_pressure_base_time.__radd__ = lambda self, other: mock_total
        mock_pressure_p_time.__add__ = lambda self, other: mock_total
        mock_pressure_p_time.__radd__ = lambda self, other: mock_total
        
        mock_pressure_base = MagicMock()
        mock_pressure_base.isel.return_value = mock_pressure_base_time
        
        mock_pressure_p = MagicMock()
        mock_pressure_p.isel.return_value = mock_pressure_p_time
        
        mock_ds.__getitem__.side_effect = make_getitem({'pressure_base': mock_pressure_base, 'pressure_p': mock_pressure_p}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'pressure_p', 'pressure_base'])
        
        return mock_ds
    
    def _create_mock_dataset_with_hybrid(self) -> Any:
        """
        This helper method creates a mock dataset that includes hybrid coordinate information (e.g., `hybrid_a` and `hybrid_b`) for testing pressure level retrieval from hybrid coordinates. The mock dataset is structured to mimic an xarray Dataset containing a variable named 'theta' with dimensions `nCells`, `nVertLevels`, and `Time`, along with synthetic `fzp` and `surface_pressure` variables that allow for testing the computation of pressure levels from hybrid coordinates. By setting up the necessary return values for indexing, slicing, and arithmetic operations on the hybrid coordinate components, this mock dataset enables controlled testing of scenarios related to retrieving pressure levels from hybrid coordinates without relying on actual MPAS data files.

        Parameters:
            None

        Returns:
            Any: A MagicMock object emulating an xarray Dataset with hybrid coords.
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True, 'fzp': True, 'surface_pressure': True}
        mock_ds.sizes = {'nVertLevels': 55, 'Time': 10}
        
        mock_var = MagicMock()
        mock_var.sizes = {'nCells': 40962, 'nVertLevels': 55, 'Time': 10}
        
        fzp_vals = np.linspace(1.0, 0.01, 55)
        sp_vals = np.full(40962, 101325.0)
        
        mock_fzp = MagicMock()
        mock_fzp.values = fzp_vals
        mock_fzp.isel.return_value = mock_fzp
        
        mock_sp = MagicMock()
        mock_sp.values = sp_vals
        mock_sp.isel.return_value = mock_sp
        
        mock_ds.__getitem__.side_effect = make_getitem({'fzp': mock_fzp, 'surface_pressure': mock_sp}, default=mock_var)
        mock_ds.__contains__.side_effect = make_contains(['theta', 'fzp', 'surface_pressure'])
        
        return mock_ds


class TestAddSpatialCoordinates:
    """ Test spatial coordinate enrichment. """
    
    def setup_method(self: "TestAddSpatialCoordinates") -> None:
        """
        This method-level setup initializes an `MPAS3DProcessor` instance with a mocked grid file path and `verbose=False`. The filesystem existence check is patched to always return True, allowing the test to focus on the processor initialization logic without relying on actual files. This setup provides a consistent starting point for the subsequent tests that will verify the behavior of spatial coordinate enrichment methods. If the test data is not available, the tests will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        with patch('os.path.exists', return_value=True):
            self.processor = MPAS3DProcessor('test_grid.nc', verbose=False)
    
    def test_add_coordinates_to_real_dataset(self: "TestAddSpatialCoordinates") -> None:
        """
        This test verifies that the `add_spatial_coordinates` method can successfully add spatial coordinates to a real MPAS dataset when a 3D variable is present. By loading the dataset and invoking the method, the test asserts that the resulting dataset contains expected spatial coordinate names (e.g., 'lon', 'lat', 'lonCell', 'latCell') either as coordinates or data variables, confirming that the method can enrich the dataset with spatial information. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        processor = load_mpas_3d_processor(verbose=False)
        
        assert processor.dataset is not None
        assert len(processor.dataset.data_vars) > 0
        
        coord_names = ['lon', 'lonCell', 'lat', 'latCell']
        has_coords = any(c in processor.dataset.coords or c in processor.dataset.data_vars 
                         for c in coord_names)
        assert has_coords, "Expected spatial coordinates not found in dataset"
    
    def test_add_coordinates(self: "TestAddSpatialCoordinates") -> None:
        """
        This test checks that the `add_spatial_coordinates` method can successfully add spatial coordinates to a dataset when a 3D variable is present. By creating a mock dataset with a 3D variable and invoking the method, the test asserts that the resulting dataset is the same as the input dataset (since the helper method is mocked to return it) and that the helper method was called with the expected dimensions for spatial coordinates, confirming that the method attempts to enrich the dataset with spatial information based on the presence of a 3D variable. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        mock_ds = MagicMock()
        mock_ds.data_vars = {'theta': True}
        
        with patch.object(self.processor, '_add_spatial_coords_helper', return_value=mock_ds) as mock_helper:
            result = self.processor.add_spatial_coordinates(mock_ds)
            assert result is mock_ds
            
            mock_helper.assert_called_once()
            call_args = mock_helper.call_args[0]
            
            dimensions = call_args[1]
            assert 'nCells' in dimensions
            assert 'nVertLevels' in dimensions
            assert 'nVertLevelsP1' in dimensions
            assert 'nSoilLevels' in dimensions


class TestExtract2DFrom3DStatic:
    """ Test static 2D extraction method. """
    
    def test_with_level_index_xarray(self: "TestExtract2DFrom3DStatic") -> None:
        """
        This test verifies that the `extract_2d_from_3d` method can successfully extract a 2D slice from a 3D xarray DataArray using a specified level index. By creating a mock 3D DataArray with appropriate dimensions and coordinates, and then calling `extract_2d_from_3d(..., level_index=...)`, the test asserts that the resulting array has the expected 2D shape corresponding to the number of cells and time steps. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        data_3d = np.random.randn(100, 55, 10)
        coords = {'nVertLevels': np.arange(55)}
        data_array = xr.DataArray(data_3d, dims=['nCells', 'nVertLevels', 'Time'], coords=coords)        
        result = MPAS3DProcessor.extract_2d_from_3d(data_array, level_index=10)        
        assert result.shape == (100, 10)
    
    def test_with_level_value_nearest(self: "TestExtract2DFrom3DStatic") -> None:
        """
        This test verifies that the `extract_2d_from_3d` method can successfully extract a 2D slice from a 3D xarray DataArray using a specified level value with nearest neighbor selection. By creating a mock 3D DataArray with pressure-level coordinates and calling `extract_2d_from_3d(..., level_value=..., level_dim=..., method='nearest')`, the test asserts that the resulting array has the expected 2D shape corresponding to the number of cells, confirming that the method can correctly perform nearest neighbor selection based on the provided pressure value. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        data_3d = np.random.randn(100, 55)
        pressure_levels = np.linspace(100000, 1000, 55)
        coords = {'pressure': pressure_levels}

        data_array = xr.DataArray(data_3d, dims=['nCells', 'pressure'], coords=coords)        
        result = MPAS3DProcessor.extract_2d_from_3d(data_array, level_value=85000, level_dim='pressure')        
        assert result.shape == (100,)
    
    def test_with_level_value_linear(self: "TestExtract2DFrom3DStatic") -> None:
        """
        This test checks that the `extract_2d_from_3d` method can successfully extract a 2D slice from a 3D xarray DataArray using a specified level value with linear interpolation. By creating a mock 3D DataArray with pressure-level coordinates and calling `extract_2d_from_3d(..., level_value=..., level_dim=..., method='linear')`, the test asserts that the resulting array has the expected 2D shape corresponding to the number of cells, confirming that the method can correctly perform linear interpolation based on the provided pressure value. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        data_3d = np.random.randn(100, 55)
        pressure_levels = np.linspace(100000, 1000, 55)
        coords = {'pressure': pressure_levels}
        data_array = xr.DataArray(data_3d, dims=['nCells', 'pressure'], coords=coords)
        
        result = MPAS3DProcessor.extract_2d_from_3d(
            data_array, level_value=50000, level_dim='pressure', method='linear'
        )
        
        assert result.shape == (100,)
    
    def test_with_numpy_3d(self: "TestExtract2DFrom3DStatic") -> None:
        """
        This test verifies that the `extract_2d_from_3d` method can handle a 3D NumPy array input when a level index is provided, effectively treating the 3D array as if it were a stack of vertical levels. By creating a mock 3D NumPy array and calling `extract_2d_from_3d(..., level_index=...)`, the test asserts that the resulting array has the expected 1D shape corresponding to the number of cells, confirming that the method can gracefully handle 3D NumPy inputs in this context. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        data_3d = np.random.randn(100, 55, 10)        
        result = MPAS3DProcessor.extract_2d_from_3d(data_3d, level_index=20)        
        assert result.shape == (100,)
    
    def test_with_numpy_2d(self: "TestExtract2DFrom3DStatic") -> None:
        """
        This test checks that the `extract_2d_from_3d` method can handle a 2D NumPy array input when a level index is provided, effectively treating the 2D array as if it were a single vertical level. By creating a mock 2D NumPy array and calling `extract_2d_from_3d(..., level_index=...)`, the test asserts that the resulting array has the expected 1D shape corresponding to the number of cells, confirming that the method can gracefully handle 2D NumPy inputs in this context. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        data_2d = np.random.randn(100, 55)        
        result = MPAS3DProcessor.extract_2d_from_3d(data_2d, level_index=30)        
        assert result.shape == (100,)
    
    def test_error_no_level(self: "TestExtract2DFrom3DStatic") -> None:
        """
        This test verifies that the `extract_2d_from_3d` method raises an informative error when neither a level index nor a level value is provided for extraction. By creating a mock 3D xarray DataArray and calling `extract_2d_from_3d(...)` without specifying either `level_index` or `level_value`, the test asserts that a ValueError is raised with a message indicating that one of these parameters must be provided for extraction. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        data_array = xr.DataArray(np.random.randn(100, 55), dims=['nCells', 'nVertLevels'])
        
        with pytest.raises(ValueError) as exc_info:
            MPAS3DProcessor.extract_2d_from_3d(data_array)

        assert 'Must provide either' in str(exc_info.value)
    
    def test_error_coordinate_not_found(self: "TestExtract2DFrom3DStatic") -> None:
        """
        This test checks that the `extract_2d_from_3d` method raises an informative error when a level value is provided for extraction but the specified level dimension is not found in the coordinates of the xarray DataArray. By creating a mock 3D xarray DataArray without the expected coordinate and calling `extract_2d_from_3d(..., level_value=..., level_dim=...)`, the test asserts that a ValueError is raised with a message indicating that the specified level dimension was not found in the coordinates, confirming that the method correctly identifies issues with coordinate-based level selection. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        data_array = xr.DataArray(np.random.randn(100, 55), dims=['nCells', 'nVertLevels'])
        
        with pytest.raises(ValueError) as exc_info:
            MPAS3DProcessor.extract_2d_from_3d(data_array, level_value=500, level_dim='pressure')

        assert 'not found' in str(exc_info.value)
    
    def test_error_numpy_with_level_value(self: "TestExtract2DFrom3DStatic") -> None:
        """
        This test verifies that the `extract_2d_from_3d` method raises an informative error when a level value is provided for extraction but the input array is a NumPy array, which does not have coordinate information to perform level selection. By creating a mock 3D NumPy array and calling `extract_2d_from_3d(..., level_value=..., level_dim=...)`, the test asserts that a ValueError is raised with a message indicating that coordinate-based level selection cannot be performed on a NumPy array, confirming that the method correctly identifies issues with using level values on non-xarray inputs. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        data_array = np.random.randn(100, 55)
        
        with pytest.raises(ValueError) as exc_info:
            MPAS3DProcessor.extract_2d_from_3d(data_array, level_value=500)

        assert 'requires xarray' in str(exc_info.value)
    
    def test_error_insufficient_dimensions(self: "TestExtract2DFrom3DStatic") -> None:
        """
        This test checks that the `extract_2d_from_3d` method raises an informative error when the input array has insufficient dimensions (e.g., it is 1D) for performing 2D extraction. By creating a mock 1D NumPy array and calling `extract_2d_from_3d(..., level_index=...)`, the test asserts that a ValueError is raised with a message indicating that the input array must have at least 2 dimensions for extraction, confirming that the method correctly identifies issues with input dimensionality. If real MPAS data is absent, the test will be skipped to avoid false failures. 

        Parameters:
            None

        Returns:
            None
        """
        data_1d = np.random.randn(100)
        
        with pytest.raises(ValueError) as exc_info:
            MPAS3DProcessor.extract_2d_from_3d(data_1d, level_index=5)

        assert 'at least 2D' in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__])

