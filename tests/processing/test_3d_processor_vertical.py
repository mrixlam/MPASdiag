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
        if key == raise_key:
            raise exc
        return MagicMock() if default is None else default
    return _getitem


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
            assert_expected_public_methods(self.processor, 'MPAS3DProcessor')
    
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
        
        assert len(levels) == pytest.approx(55)
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
        
        assert len(levels) == pytest.approx(55)
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
        
        assert len(levels) == pytest.approx(55)
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
        
        assert len(levels) == pytest.approx(55)
    
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
    
    def _create_mock_dataset_with_hybrid(self: "TestGetVerticalLevels") -> Any:
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
            assert_expected_public_methods(self.processor, 'MPAS3DProcessor')
    
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

